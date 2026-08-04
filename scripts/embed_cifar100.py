#!/usr/bin/env python3
"""Embed CIFAR-100 train set with DINOv2, save embeddings cache for downstream
silhouette / Calinski-Harabasz / Davies-Bouldin scan via select_optimal_k.py.

Parallels the NLP side: BGE-large embeddings of 195K SlimPajama chunks →
silhouette ≈ 0.03 (no discrete structure). Expectation for CIFAR-100: image
embeddings should have much higher silhouette (0.1-0.4+) because visual
categories are inherently discrete — if confirmed, this cross-modal comparison
supports the paper narrative that SpecDrop's PPL plateau on NLP reflects
embedding-space smoothness, not a method defect.

Uses HuggingFace `facebook/dinov2-base` (86M, 768-dim) — modern SSL vision
embedder, same philosophical choice as BGE-large on the NLP side (strong,
mainstream). Swappable via --embedder.

CIFAR-100 is 32×32, DINOv2 expects 14-patch multiples (default 224×224); we
upsample via bicubic.

Output schema matches data/cluster_chunks.py's embedding cache:
    {'embeddings': (N, D) float32 tensor,
     'labels':     (N,)   long   tensor   # fine class labels [0, 99]
     'num_chunks': N,
     'dim':        D,
     'embedder':   str,
     'dataset':    'cifar100_train'}

Usage:
    python scripts/embed_cifar100.py \\
        --data-dir data_cache/cifar-100-python \\
        --output data_cache/cifar100_embeddings_dinov2-base.pt \\
        --embedder facebook/dinov2-base \\
        --device cuda
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100


def embed(data_dir, output_path, embedder='facebook/dinov2-base',
          batch_size=128, device='cuda', num_workers=4, image_size=224):
    if os.path.exists(output_path):
        print(f'[embed-cifar] output already exists at {output_path}, skipping')
        return

    # torchvision's CIFAR100 expects root to be the directory CONTAINING
    # cifar-100-python/. Accept either form ergonomically.
    root = data_dir
    if os.path.basename(os.path.normpath(data_dir)) == 'cifar-100-python':
        root = os.path.dirname(os.path.normpath(data_dir))
        print(f'[embed-cifar] stripping trailing cifar-100-python/ → root={root}',
              flush=True)

    print(f'[embed-cifar] loading CIFAR-100 from {root}...', flush=True)
    tfm = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    ds = CIFAR100(root=root, train=True, download=False, transform=tfm)
    print(f'[embed-cifar] {len(ds):,} train images, {image_size}×{image_size} after resize',
          flush=True)

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=(device == 'cuda'))

    print(f'[embed-cifar] loading {embedder}...', flush=True)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(embedder).to(device).eval()
    # Pooled feature dim — for DINOv2 it's the [CLS] hidden size.
    dim = getattr(model.config, 'hidden_size', None)

    print(f'[embed-cifar] embedding {len(ds):,} images with '
          f'batch_size={batch_size} on {device}...', flush=True)
    all_feats = []
    all_labels = []
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, lbl) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            # DINOv2's forward returns ModelOutput with last_hidden_state (B, T, D)
            # where T includes the [CLS] token at index 0. Use the CLS embedding.
            out = model(pixel_values=imgs)
            cls = out.last_hidden_state[:, 0]  # (B, D) — CLS token
            all_feats.append(cls.cpu())
            all_labels.append(lbl)
            if (i + 1) % 10 == 0 or i == 0:
                done = (i + 1) * batch_size
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f'  [{min(done, len(ds)):>5}/{len(ds)}] {rate:.0f} img/s',
                      flush=True)
    emb = torch.cat(all_feats, 0).float()  # (N, D)
    labels = torch.cat(all_labels, 0).long()
    dur = time.time() - t0
    print(f'[embed-cifar] done in {dur:.1f}s, shape={tuple(emb.shape)}',
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path + '.tmp'
    torch.save({
        'embeddings': emb,
        'labels': labels,
        'num_chunks': int(emb.shape[0]),
        'dim': int(emb.shape[1]),
        'embedder': embedder,
        'dataset': 'cifar100_train',
        'image_size': image_size,
    }, tmp_path)
    os.replace(tmp_path, output_path)
    print(f'[embed-cifar] saved {emb.shape} to {output_path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data_cache/cifar-100-python',
                    help='Directory containing CIFAR-100 train/test/meta files')
    p.add_argument('--output', required=True,
                    help='Output .pt path for the embedding cache')
    p.add_argument('--embedder', default='facebook/dinov2-base',
                    help='HuggingFace model id (default: facebook/dinov2-base, '
                         '86M params 768-dim). Alternatives: facebook/dinov2-small, '
                         'facebook/dinov2-large, facebook/dinov2-giant.')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--device', default='cuda', help="'cuda', 'cpu', or 'mps'")
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--image-size', type=int, default=224,
                    help='Resize CIFAR 32×32 up to this size (default 224, '
                         'standard for ImageNet-pretrained models)')
    args = p.parse_args()

    if not os.path.exists(args.data_dir):
        print(f'error: CIFAR-100 data dir not found: {args.data_dir}',
              file=sys.stderr); sys.exit(1)

    embed(args.data_dir, args.output, embedder=args.embedder,
          batch_size=args.batch_size, device=args.device,
          num_workers=args.num_workers, image_size=args.image_size)


if __name__ == '__main__':
    main()
