#!/usr/bin/env python3
"""Fine-tune a trained dense checkpoint into a coarse-category classifier
(dedicated pre-classifier evaluation, paper App. E.4).

Loads the dense faithful checkpoint, swaps the classification head in place
(CIFAR 100->20, ImageNet 1000->46), and trains on the coarse label
(batch[2]). No mixup/cutmix (avoids the MixupCutmix num_classes landmine and
is counterproductive for short fine-tunes). Fixed schedule, no early
stopping; the test split is touched exactly once, at the end, to report
coarse accuracy (avoids any tuned-on-test complaint).

Usage:
    python scripts/finetune_coarse_classifier.py --setting cifar --seed 42
    python scripts/finetune_coarse_classifier.py --setting vit --seed 123 \
        --epochs 3 --freeze_backbone
Output: {out_dir}/best.pt (key model_state_dict) + results.json (config with
updated num_classes so eval_predicted_cluster's _load_cfg tier-2 resolves it).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

DEFAULTS = {
    'cifar': dict(src='outputs/rtx5090_cifar100_faithful/resnet110_s{seed}',
                  out='outputs/coarse_clf_cifar/resnet110_coarse_s{seed}',
                  n_coarse=20, epochs=15, lr=0.01, head_lr=0.1),
    'vit':   dict(src='outputs/rtx5090_imagenet_vit_faithful/vit_small_s{seed}',
                  out='outputs/coarse_clf_vit/vit_small_coarse_s{seed}',
                  n_coarse=46, epochs=3, lr=2e-5, head_lr=2e-4),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--setting', choices=('cifar', 'vit'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--src_run_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--freeze_backbone', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    d = DEFAULTS[args.setting]
    src = args.src_run_dir or d['src'].format(seed=args.seed)
    out = args.out_dir or d['out'].format(seed=args.seed)
    if args.freeze_backbone:
        out += '_probe'
    epochs = args.epochs or d['epochs']
    lr = args.lr or d['lr']
    if os.path.exists(f'{out}/results.json'):
        print(f'[coarse_clf] {out} exists, skipping')
        return
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    from scripts.eval_predicted_cluster import _load_cfg, _load_state_dict
    cfg = _load_cfg(src)
    from models import build_model
    model = build_model(cfg).to(args.device)
    _load_state_dict(model, f'{src}/best.pt', args.device)

    # In-place head swap AFTER loading (sidesteps strict-load shape mismatch)
    K = d['n_coarse']
    if args.setting == 'cifar':
        model.fc = nn.Linear(model.fc.in_features, K).to(args.device)
        head = model.fc
    else:
        model.head = nn.Linear(model.head.in_features, K).to(args.device)
        head = model.head

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

    if args.setting == 'cifar':
        from data.cifar100 import get_dataloaders
        train_loader, test_loader = get_dataloaders(
            data_dir='./data_cache', batch_size=128, num_workers=8,
            device=args.device)
        backbone_params = [p for n, p in model.named_parameters()
                          if not n.startswith('fc') and p.requires_grad]
        opt = torch.optim.SGD(
            [{'params': backbone_params, 'lr': lr},
             {'params': head.parameters(), 'lr': d['head_lr']}],
            momentum=0.9, weight_decay=5e-4)
    else:
        from data.imagenet import get_imagenet_dataloaders
        train_loader, test_loader, _ = get_imagenet_dataloaders(
            data_dir='./data_cache/imagenet', batch_size=256, num_workers=16,
            augmentation='basic', prefetch_factor=4)
        backbone_params = [p for n, p in model.named_parameters()
                          if not n.startswith('head') and p.requires_grad]
        opt = torch.optim.AdamW(
            [{'params': backbone_params, 'lr': lr},
             {'params': head.parameters(), 'lr': d['head_lr']}],
            weight_decay=0.05)

    crit = nn.CrossEntropyLoss()
    total_steps = epochs * len(train_loader)
    step = 0
    model.train()
    use_amp = args.device.startswith('cuda')
    for ep in range(epochs):
        run_loss = n_seen = 0
        for batch in train_loader:
            x, coarse = batch[0].to(args.device, non_blocking=True), \
                        batch[2].to(args.device, non_blocking=True)
            for g in opt.param_groups:
                g['lr'] = g.setdefault('_base', g['lr']) * 0.5 * (
                    1 + math.cos(math.pi * step / total_steps))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = crit(model(x), coarse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            step += 1
        print(f'[coarse_clf][{args.setting} s{args.seed}] epoch {ep+1}/{epochs} '
              f'train_loss={run_loss/max(1,n_seen):.4f}', flush=True)

    # Single final pass on the test split (report-only, no model selection)
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16,
                                         enabled=use_amp):
        for batch in test_loader:
            x, coarse = batch[0].to(args.device), batch[2].to(args.device)
            correct += (model(x).argmax(-1) == coarse).sum().item()
            total += coarse.numel()
    acc = 100.0 * correct / max(1, total)

    cfg_out = json.loads(json.dumps(cfg))
    cfg_out['model']['num_classes'] = K
    cfg_out['output_dir'] = out
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epochs,
                'best_acc': acc}, f'{out}/best.pt')
    with open(f'{out}/results.json', 'w') as f:
        json.dump({'config': cfg_out, 'best_top1': acc,
                   'coarse_acc': acc, 'src_run_dir': src,
                   'freeze_backbone': args.freeze_backbone,
                   'epochs': epochs, 'lr': lr}, f, indent=2)
    print(f'[coarse_clf][{args.setting} s{args.seed}] COARSE ACC = {acc:.2f} '
          f'-> {out}')


if __name__ == '__main__':
    main()
