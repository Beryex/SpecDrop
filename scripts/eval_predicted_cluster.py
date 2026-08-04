#!/usr/bin/env python3
"""E1 — Predicted-category routing (label-leak control).

For each setting (CIFAR-100 K=20 / ViT ImageNet K=46), run ours' trained
checkpoint at inference using a PREDICTED cluster_id (from an independent
dense classifier) instead of the ground-truth coarse label. This tests
deployability when oracle metadata is unavailable.

The "predictor" is the existing dense baseline (ResNet-110 for CIFAR,
ViT-Small for ImageNet) whose argmax fine-class prediction maps
deterministically to a coarse cluster_id via the official
fine→coarse mapping. No new training required: the dense classifier was
trained independently of SpecDrop and predicts fine-class without seeing
coarse labels at test time.

The output JSON enables a 3-way comparison against existing P0:
  oracle Δ (ours with ground-truth):    `outputs/rtx5090_*_faithful/ours_*/results.json`
  predicted Δ (ours with classifier):   THIS SCRIPT
  uniform Δ (mech-OFF fallback):        `outputs/eval_uniform_mask/<setting>_s<seed>.json`

Usage:
    python scripts/eval_predicted_cluster.py --setting cifar --seed 42
    python scripts/eval_predicted_cluster.py --setting vit   --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml


def _load_cfg(run_dir: str) -> dict:
    """3-tier fallback (mirrors eval_uniform_mask.py)."""
    import glob
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml'))
    if cands:
        with open(cands[0]) as f:
            return yaml.safe_load(f)
    rj = f'{run_dir}/results.json'
    if os.path.exists(rj):
        with open(rj) as f:
            r = json.load(f)
        if 'config' in r:
            return r['config']
    raise FileNotFoundError(
        f'no recoverable config for {run_dir}: tried _tmp*.yaml + results.json::config')


def _load_src_metric(run_dir: str, key: str) -> float | None:
    p = f'{run_dir}/results.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = json.load(f)
    return r.get(key)


def _load_state_dict(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def _build_trained_algorithm(cfg, K: int, M: int):
    """Reproduce ours' trained mask state at terminal warmup progress."""
    from algorithms import build_algorithm
    cfg = dict(cfg)
    cfg.setdefault('algorithm', {})
    # Inject runtime M for compatibility with existing build_algorithm.
    cfg['algorithm']['_num_categories'] = M
    cfg['model'] = dict(cfg['model'])
    cfg['model']['num_branches'] = K
    algo = build_algorithm(cfg)
    if algo is None:
        raise ValueError('built algorithm is None — check cfg.algorithm.type')
    # Advance any warmup state to terminal (step + epoch modes)
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    if hasattr(algo, 'warmup_unit'):
        total_epochs = cfg.get('training', {}).get('epochs', 200)
        advance_softspecdrop_to_terminal(algo, total_epochs)
    return algo


# ─── Fine → coarse mapping ─────────────────────────────────────────────────
def _cifar100_fine_to_coarse() -> torch.Tensor:
    from data.cifar100 import FINE_TO_COARSE
    return torch.tensor(FINE_TO_COARSE, dtype=torch.long)


def _imagenet_fine_to_coarse(breeds_dir: str = None) -> torch.Tensor:
    """1000-elem tensor: fine_to_coarse[fine_idx] = coarse_idx (BREEDS K=46)."""
    from data.imagenet import _build_breeds_mapping, BREEDS_HIERARCHY_DIR
    breeds_dir = breeds_dir or BREEDS_HIERARCHY_DIR
    mapping, num_super, _ = _build_breeds_mapping(breeds_dir=breeds_dir, T=60, C=10)
    out = torch.full((1000,), -1, dtype=torch.long)
    for fine_idx, coarse_idx in mapping.items():
        out[fine_idx] = coarse_idx
    if (out < 0).any():
        # Fill any unmapped fine class to a "miscellaneous" id (max+1) — should
        # not happen under canonical T=60 C=10 but guard anyway.
        unmapped = (out < 0).sum().item()
        print(f'[predicted_cluster] WARN: {unmapped} ImageNet classes unmapped; '
              f'assigning to miscellaneous id={num_super}')
        out[out < 0] = num_super
    return out


# ─── CIFAR ──────────────────────────────────────────────────────────────────
def eval_cifar(seed: int, device: str,
                predictor_run_dir: str | None = None) -> dict:
    """Evaluate ours_cifar with predicted superclass from dense ResNet-110."""
    ours_run_dir = f'outputs/rtx5090_cifar100_faithful/ours_s{seed}'
    pred_run_dir = predictor_run_dir or f'outputs/rtx5090_cifar100_faithful/resnet110_s{seed}'

    ours_cfg = _load_cfg(ours_run_dir)
    pred_cfg = _load_cfg(pred_run_dir)
    src_top1 = _load_src_metric(ours_run_dir, 'best_top1')
    if src_top1 is not None and src_top1 <= 1.0:
        src_top1 *= 100

    from models import build_model
    ours_model = build_model(ours_cfg).to(device)
    pred_model = build_model(pred_cfg).to(device)
    _load_state_dict(ours_model, f'{ours_run_dir}/best.pt', device)
    _load_state_dict(pred_model, f'{pred_run_dir}/best.pt', device)

    from scripts._diag_helpers import require_keys
    require_keys(ours_cfg['model'], ('num_branches',), f'cfg["model"] in {ours_run_dir}')
    K = ours_cfg['model']['num_branches']
    # CIFAR-100 has 20 superclasses by dataset construction (data.cifar100.
    # NUM_SUPERCLASSES); the cfg may set num_categories under `algorithm:`
    # (canonical) or `data:` (legacy). Try both, fall back to K (CIFAR ours
    # is K==M==20). Don't require_keys('data', 'num_categories') because the
    # rtx5090_cifar100_faithful configs don't put it under data.
    M = (ours_cfg.get('algorithm', {}).get('num_categories')
         or ours_cfg.get('data', {}).get('num_categories', K))
    algo = _build_trained_algorithm(ours_cfg, K, M)
    # ViT/CIFAR multi-branch models expose `mask_scale` as a plain attr; some
    # variants also provide a `set_mask_scale()` setter. Match eval_uniform_mask.py
    # fallback chain so both paths are covered (else mask_scale stays None and
    # forward raises "branch_mask provided but self.mask_scale is None").
    if hasattr(ours_model, 'set_mask_scale'):
        ours_model.set_mask_scale(algo.expected_mask_sum)
    elif hasattr(ours_model, 'mask_scale'):
        ours_model.mask_scale = algo.expected_mask_sum

    fine_to_coarse = _cifar100_fine_to_coarse().to(device)

    from data.cifar100 import get_dataloaders
    dcfg = ours_cfg['data']
    _, val_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=ours_cfg['training'].get('batch_size', 128),
        num_workers=0, device=device,
    )

    correct = 0; total = 0
    coarse_correct = 0  # diagnostic: predictor's coarse accuracy
    with torch.no_grad():
        for batch in val_loader:
            x, fine_y, coarse_y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            # Predict fine class via dense baseline
            try:
                pred_fine_logits = pred_model(x)
            except TypeError:
                # Some MultiBranch models also accept branch_mask kwarg; try without
                pred_fine_logits = pred_model(x, branch_mask=None)
            pred_fine = pred_fine_logits.argmax(dim=-1)
            pred_coarse = fine_to_coarse[pred_fine]
            coarse_correct += (pred_coarse == coarse_y).sum().item()
            # Use predicted coarse as cluster_id for ours
            mask = algo.get_mask(pred_coarse, training=False).to(device)
            try:
                logits = ours_model(x, branch_mask=mask)
            except TypeError:
                logits = ours_model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == fine_y).sum().item()
            total += fine_y.numel()
    top1 = 100.0 * correct / max(1, total)
    pred_coarse_acc = 100.0 * coarse_correct / max(1, total)
    delta = top1 - (src_top1 or 0.0)

    print(f'[predicted_cluster][cifar s{seed}] '
          f'src(oracle)={src_top1:.2f}  predicted={top1:.2f}  Δ={delta:+.2f}  '
          f'(predictor coarse-acc={pred_coarse_acc:.2f}%)')
    return {
        'setting': 'cifar', 'seed': seed,
        'src_run_dir': ours_run_dir, 'predictor_run_dir': pred_run_dir,
        'src_metric': src_top1, 'predicted_cluster_metric': top1,
        'delta': delta, 'predictor_coarse_acc': pred_coarse_acc,
        'metric_name': 'top1_acc',
    }


# ─── ViT ImageNet ───────────────────────────────────────────────────────────
def eval_vit(seed: int, device: str,
              predictor_run_dir: str | None = None) -> dict:
    """Evaluate ours_vit with predicted BREEDS-46 cluster from dense ViT-Small."""
    ours_run_dir = f'outputs/rtx5090_imagenet_vit_faithful/ours_vit_s{seed}'
    pred_run_dir = predictor_run_dir or f'outputs/rtx5090_imagenet_vit_faithful/vit_small_s{seed}'

    ours_cfg = _load_cfg(ours_run_dir)
    pred_cfg = _load_cfg(pred_run_dir)
    src_top1 = _load_src_metric(ours_run_dir, 'best_top1')
    if src_top1 is not None and src_top1 <= 1.0:
        src_top1 *= 100

    from models import build_model
    ours_model = build_model(ours_cfg).to(device)
    pred_model = build_model(pred_cfg).to(device)
    _load_state_dict(ours_model, f'{ours_run_dir}/best.pt', device)
    _load_state_dict(pred_model, f'{pred_run_dir}/best.pt', device)

    from scripts._diag_helpers import require_keys
    require_keys(ours_cfg['model'], ('num_branches',), f'cfg["model"] in {ours_run_dir}')
    K = ours_cfg['model']['num_branches']
    algo = _build_trained_algorithm(ours_cfg, K, K)
    # ViT/CIFAR multi-branch models expose `mask_scale` as a plain attr; some
    # variants also provide a `set_mask_scale()` setter. Match eval_uniform_mask.py
    # fallback chain so both paths are covered (else mask_scale stays None and
    # forward raises "branch_mask provided but self.mask_scale is None").
    if hasattr(ours_model, 'set_mask_scale'):
        ours_model.set_mask_scale(algo.expected_mask_sum)
    elif hasattr(ours_model, 'mask_scale'):
        ours_model.mask_scale = algo.expected_mask_sum

    fine_to_coarse = _imagenet_fine_to_coarse().to(device)

    from data.imagenet import get_imagenet_dataloaders
    dcfg = ours_cfg['data']
    _, val_loader, _ = get_imagenet_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache/imagenet'),
        batch_size=ours_cfg['training'].get('batch_size', 256),
        num_workers=8, augmentation='basic', prefetch_factor=2,
    )

    correct = 0; total = 0
    coarse_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            fine_y = batch[1].to(device, non_blocking=True)
            coarse_y = batch[2].to(device, non_blocking=True)
            try:
                pred_fine_logits = pred_model(x)
            except TypeError:
                pred_fine_logits = pred_model(x, branch_mask=None)
            pred_fine = pred_fine_logits.argmax(dim=-1)
            pred_coarse = fine_to_coarse[pred_fine]
            coarse_correct += (pred_coarse == coarse_y).sum().item()
            mask = algo.get_mask(pred_coarse, training=False).to(device)
            try:
                logits = ours_model(x, branch_mask=mask)
            except TypeError:
                logits = ours_model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == fine_y).sum().item()
            total += fine_y.numel()
    top1 = 100.0 * correct / max(1, total)
    pred_coarse_acc = 100.0 * coarse_correct / max(1, total)
    delta = top1 - (src_top1 or 0.0)

    print(f'[predicted_cluster][vit s{seed}] '
          f'src(oracle)={src_top1:.2f}  predicted={top1:.2f}  Δ={delta:+.2f}  '
          f'(predictor coarse-acc={pred_coarse_acc:.2f}%)')
    return {
        'setting': 'vit', 'seed': seed,
        'src_run_dir': ours_run_dir, 'predictor_run_dir': pred_run_dir,
        'src_metric': src_top1, 'predicted_cluster_metric': top1,
        'delta': delta, 'predictor_coarse_acc': pred_coarse_acc,
        'metric_name': 'top1_acc',
    }


# ─── ViT CLIP zero-shot predictor (E-CLIP-1) ───────────────────────────────
def _get_breeds_supercategory_names() -> list:
    """46 BREEDS supercategory names from data.imagenet._build_breeds_mapping."""
    from data.imagenet import _build_breeds_mapping, BREEDS_HIERARCHY_DIR
    _, _, names = _build_breeds_mapping(breeds_dir=BREEDS_HIERARCHY_DIR, T=60, C=10)
    return names


def _clip_features_from_output(out, projection):
    """Bypass `get_text/image_features` whose return-type drifted across
    transformers v4/v5 (sometimes Tensor, sometimes BaseModelOutputWithPooling).
    Take pooled hidden state from the underlying model output and project
    through the corresponding projection head — canonical CLIP path."""
    if isinstance(out, torch.Tensor):
        return projection(out) if out.dim() == 2 else projection(out[:, 0])
    pooled = getattr(out, 'pooler_output', None)
    if pooled is None and hasattr(out, 'last_hidden_state'):
        pooled = out.last_hidden_state[:, 0]  # CLS / EOS fallback
    if pooled is None:
        raise RuntimeError(f'cannot extract pooled features from {type(out)}')
    return projection(pooled)


def _build_clip_breeds_predictor(device: str):
    """Returns (clip_model, text_features (46, D), breeds_names (list of 46)).

    Bypasses CLIPModel.get_text_features() — it returns BaseModelOutputWithPooling
    on transformers v5+ instead of Tensor. We call text_model + text_projection
    directly, which is the canonical CLIP zero-shot computation.
    """
    from transformers import CLIPModel, CLIPProcessor
    print('[predicted_cluster] loading CLIP-ViT-B/16 ...')
    clip_id = 'openai/clip-vit-base-patch16'
    clip = CLIPModel.from_pretrained(clip_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_id)
    breeds_names = _get_breeds_supercategory_names()
    prompts = [f'a photo of a {n}' for n in breeds_names]
    text_inputs = processor(text=prompts, return_tensors='pt', padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_out = clip.text_model(**text_inputs)
        text_features = _clip_features_from_output(text_out, clip.text_projection)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f'[predicted_cluster] CLIP text features built: shape={tuple(text_features.shape)} '
          f'(K=46 BREEDS supercategories)')
    return clip, text_features, breeds_names


# CLIP / ImageNet preprocessing constants — used to convert ImageNet-normalized
# tensors (from existing val_loader) to CLIP-normalized tensors WITHOUT
# re-loading raw images. Mathematically equivalent to running CLIP processor.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _imagenet_norm_to_clip_norm(x: torch.Tensor) -> torch.Tensor:
    """In-place algebra: x_clip = (x_imnet * imnet_std + imnet_mean - clip_mean) / clip_std.

    Assumes x is (B, 3, H, W) ImageNet-normalized. CLIP expects 224x224; if the
    source loader yields a different size, caller should resize first.
    """
    dev = x.device
    in_mean = torch.tensor(_IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    in_std = torch.tensor(_IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    cl_mean = torch.tensor(_CLIP_MEAN, device=dev).view(1, 3, 1, 1)
    cl_std = torch.tensor(_CLIP_STD, device=dev).view(1, 3, 1, 1)
    raw = x * in_std + in_mean  # back to [0, 1]
    return (raw - cl_mean) / cl_std


def eval_vit_clip(seed: int, device: str,
                   predictor_run_dir: str | None = None) -> dict:
    """Evaluate ours_vit with CLIP-zero-shot BREEDS-46 predictor (E-CLIP-1).

    Pipeline: image → CLIP-ViT-B/16 image features → cosine sim with 46
    BREEDS-supercategory text features → argmax = predicted cluster_id →
    ours's trained SoftSpecDrop forward. NO ImageNet-trained classifier in
    the path, addressing the deployable-predictor question ("a predictor
    without label-derived metadata").

    `predictor_run_dir` ignored (CLIP weights are HuggingFace-cached).
    """
    ours_run_dir = f'outputs/rtx5090_imagenet_vit_faithful/ours_vit_s{seed}'
    ours_cfg = _load_cfg(ours_run_dir)
    src_top1 = _load_src_metric(ours_run_dir, 'best_top1')
    if src_top1 is not None and src_top1 <= 1.0:
        src_top1 *= 100

    from models import build_model
    ours_model = build_model(ours_cfg).to(device)
    _load_state_dict(ours_model, f'{ours_run_dir}/best.pt', device)

    from scripts._diag_helpers import require_keys
    require_keys(ours_cfg['model'], ('num_branches',), f'cfg["model"] in {ours_run_dir}')
    K = ours_cfg['model']['num_branches']
    algo = _build_trained_algorithm(ours_cfg, K, K)
    if hasattr(ours_model, 'set_mask_scale'):
        ours_model.set_mask_scale(algo.expected_mask_sum)
    elif hasattr(ours_model, 'mask_scale'):
        ours_model.mask_scale = algo.expected_mask_sum

    clip, text_features, breeds_names = _build_clip_breeds_predictor(device)

    from data.imagenet import get_imagenet_dataloaders
    dcfg = ours_cfg['data']
    _, val_loader, _ = get_imagenet_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache/imagenet'),
        batch_size=ours_cfg['training'].get('batch_size', 256),
        num_workers=8, augmentation='basic', prefetch_factor=2,
    )

    correct = 0; total = 0
    coarse_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            fine_y = batch[1].to(device, non_blocking=True)
            coarse_y = batch[2].to(device, non_blocking=True)

            # CLIP zero-shot BREEDS prediction (bypass get_image_features — same
            # ModelOutput return-type drift as get_text_features in transformers v5).
            x_clip = _imagenet_norm_to_clip_norm(x)
            img_out = clip.vision_model(pixel_values=x_clip)
            img_features = _clip_features_from_output(img_out, clip.visual_projection)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sim = img_features @ text_features.T  # (B, 46)
            pred_coarse = sim.argmax(dim=-1)
            coarse_correct += (pred_coarse == coarse_y).sum().item()

            # Use CLIP-predicted coarse as cluster_id for ours
            mask = algo.get_mask(pred_coarse, training=False).to(device)
            try:
                logits = ours_model(x, branch_mask=mask)
            except TypeError:
                logits = ours_model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == fine_y).sum().item()
            total += fine_y.numel()
    top1 = 100.0 * correct / max(1, total)
    pred_coarse_acc = 100.0 * coarse_correct / max(1, total)
    delta = top1 - (src_top1 or 0.0)

    print(f'[predicted_cluster][vit_clip s{seed}] '
          f'src(oracle)={src_top1:.2f}  clip_predicted={top1:.2f}  Δ={delta:+.2f}  '
          f'(CLIP coarse-acc={pred_coarse_acc:.2f}%)')
    return {
        'setting': 'vit_clip', 'seed': seed,
        'src_run_dir': ours_run_dir,
        'predictor_run_dir': 'openai/clip-vit-base-patch16 (HF cache)',
        'src_metric': src_top1, 'predicted_cluster_metric': top1,
        'delta': delta, 'predictor_coarse_acc': pred_coarse_acc,
        'metric_name': 'top1_acc',
        'predictor_type': 'clip_zero_shot_breeds46',
    }


SETTING_DISPATCH = {'cifar': eval_cifar, 'vit': eval_vit, 'vit_clip': eval_vit_clip}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--setting', required=True, choices=list(SETTING_DISPATCH))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--predictor_run_dir', default=None,
                     help='Override predictor checkpoint dir. Default: dense '
                          'baseline of the same seed. Ignored for vit_clip '
                          '(uses HF-cached CLIP weights).')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    fn = SETTING_DISPATCH[args.setting]
    payload = fn(args.seed, device, predictor_run_dir=args.predictor_run_dir)

    out_path = (args.out_json or
                 f'outputs/eval_predicted_cluster/{args.setting}_s{args.seed}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[predicted_cluster] wrote {out_path}')


if __name__ == '__main__':
    main()
