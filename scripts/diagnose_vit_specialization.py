#!/usr/bin/env python3
"""Per-BREEDS-supercategory × per-branch top1 drop when each branch is zero-ablated.

Mirror of `diagnose_nlp_specialization.py` for ViT ImageNet (K=46 BREEDS).
Output: 46×46 Δ matrix → row = supercategory, col = ablated branch idx.
Diagonal argmax (branch k hurts supercat k most) is the specialization signal.

Usage:
    python scripts/diagnose_vit_specialization.py \
        --run_dir outputs/rtx5090_imagenet_vit_faithful/ours_vit_s42

Single-seed enough — the output is a heatmap (figure), not a statistical claim.
~3h on 1 RTX 5090 (47 forward-eval passes × ~3 min each on 50K val set).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import yaml


def _load_cfg(run_dir):
    """3-tier fallback (mirror of scripts/eval_uniform_mask.py::_load_cfg):
    cell-local _tmp.yaml → results.json::config → parent _tmp_s{seed}.yaml.
    ViT writes yaml to PARENT dir (CIFAR/ViT/NLP convention) so cell dir
    only contains best.pt + results.json — fall back to one of those.
    """
    import glob
    import json
    # 1. Cell-local _tmp*.yaml (LoRA convention; not present for ViT but cheap)
    cands = sorted(glob.glob(f'{run_dir}/_tmp*.yaml'))
    if cands:
        with open(cands[0]) as f:
            return yaml.safe_load(f)
    # 2. results.json::config (CIFAR/ViT/NLP — trainer.finalize_results embeds cfg)
    rj = f'{run_dir}/results.json'
    if os.path.exists(rj):
        with open(rj) as f:
            r = json.load(f)
        if 'config' in r:
            return r['config']
    # 3. Parent _tmp_s{seed}.yaml (ViT writes here; may be stale across methods
    # but fine for single-seed diagnostics on the most-recent ours run).
    seed_match = run_dir.rstrip('/').rsplit('_s', 1)
    if len(seed_match) == 2:
        seed = seed_match[1]
        parent = os.path.dirname(run_dir)
        parent_yaml = f'{parent}/_tmp_s{seed}.yaml'
        if os.path.exists(parent_yaml):
            with open(parent_yaml) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f'no recoverable config for {run_dir}: tried '
        f'(1) _tmp*.yaml, (2) results.json::config, (3) parent _tmp_s{{seed}}.yaml')


def load(run_dir, device):
    cfg = _load_cfg(run_dir)
    from models import build_model
    from algorithms import build_algorithm
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    model = build_model(cfg).to(device)
    algorithm = build_algorithm(cfg)
    # Advance BOTH current_epoch and current_step so progress=1.0 under any
    # warmup_unit (epoch / step). ViT ours uses epoch-mode but defensive vs
    # future configs. See scripts/_diag_helpers.py.
    advance_softspecdrop_to_terminal(algorithm,
                                       cfg['training'].get('epochs', 100))
    if hasattr(model, 'mask_scale') and algorithm is not None:
        model.mask_scale = algorithm.expected_mask_sum
    ckpt = torch.load(f'{run_dir}/best.pt', map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict') or ckpt.get('state_dict') or ckpt
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return cfg, model, algorithm


def _is_specdrop_family(algorithm):
    if algorithm is None:
        return False
    return type(algorithm).__name__ in ('SoftSpecDrop', 'NoDropout', 'HardCategory')


@torch.no_grad()
def eval_per_supercat(model, loader, algorithm, device, ablate_branch, max_batches):
    """Return {supercat_id: top1_acc}. If ablate_branch is not None, zero
    that branch's contribution.

    Routing-family dispatch:
      - SpecDrop family (algorithm = SoftSpecDrop/NoDropout/HardCategory):
        zero column k in algorithm.get_mask(...) and pass via branch_mask kwarg.
      - Learned routers / native MoE (Soft MoE, Mod-Squad, ...):
        install ablation hooks via scripts/_ablation_hooks.install_ablation;
        forward without branch_mask kwarg.
    """
    use_mask = _is_specdrop_family(algorithm)
    if not use_mask and ablate_branch is not None:
        from scripts._ablation_hooks import install_ablation
        install_ablation(model, k=ablate_branch)
    try:
        correct = defaultdict(int)
        total = defaultdict(int)
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            if len(batch) >= 3:
                x, y, c = batch[:3]
            else:
                x, y = batch[:2]; c = torch.zeros(x.size(0), dtype=torch.long)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            if use_mask:
                mask = algorithm.get_mask(c, training=False).clone()
                if ablate_branch is not None:
                    mask[:, ablate_branch] = 0.0
                try:
                    logits = model(x, branch_mask=mask)
                except TypeError:
                    logits = model(x)
            else:
                logits = model(x)
            # Soft MoE / Mod-Squad / COMET ViT return (logits, aux_loss);
            # SpecDrop-family ViT returns a single tensor. Unwrap.
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = logits.argmax(dim=-1)
            for b in range(x.size(0)):
                cat = c[b].item()
                correct[cat] += int(pred[b].item() == y[b].item())
                total[cat] += 1
        return {cat: 100.0 * correct[cat] / max(1, total[cat]) for cat in total}
    finally:
        if not use_mask and ablate_branch is not None:
            from scripts._ablation_hooks import uninstall_ablation
            uninstall_ablation(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--max_batches', type=int, default=200,
                        help='val batches per pass (K+1 passes total). 200×bs=51200 ≈ 50K val.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--out_json', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Diag-ViT] run={args.run_dir}  device={device}  max_batches={args.max_batches}')
    cfg, model, algorithm = load(args.run_dir, device)
    # K determination: SpecDrop family uses cfg.model.num_branches; learned
    # routers (Soft MoE, Mod-Squad) expose K via num_experts on their FFN
    # — use scripts._ablation_hooks.model_branch_count to auto-detect.
    from scripts._ablation_hooks import model_branch_count
    K = (cfg['model'].get('num_branches')
         or cfg['model'].get('num_experts')
         or cfg['model'].get('num_slots'))
    if K is None:
        K = model_branch_count(model)
    if not K:
        raise RuntimeError(f'cannot determine K for {args.run_dir}')
    print(f'[Diag-ViT] K={K}  algorithm={cfg.get("algorithm", {}).get("type")}  '
          f'pa={cfg.get("algorithm", {}).get("p_active", "?")}  '
          f'SE_dim={cfg.get("model", {}).get("shared_expert_dim", 0)}')

    from data.imagenet import get_imagenet_dataloaders
    dcfg = cfg['data']
    _, val_loader, _ = get_imagenet_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache/imagenet'),
        batch_size=cfg['training'].get('batch_size', 256),
        num_workers=8, augmentation='basic', prefetch_factor=2,
    )

    print(f'[1/2] Baseline (no ablation)...', flush=True)
    baseline = eval_per_supercat(model, val_loader, algorithm, device,
                                  ablate_branch=None, max_batches=args.max_batches)

    print(f'[2/2] Ablating each of {K} branches...', flush=True)
    ablated = {}
    for k in range(K):
        print(f'   branch {k}/{K-1}...', flush=True)
        ablated[k] = eval_per_supercat(model, val_loader, algorithm, device,
                                        ablate_branch=k, max_batches=args.max_batches)

    # Build M × K Δ matrix: row supercat, col branch, Δ = ablated - baseline
    cats = sorted(baseline.keys())
    M = len(cats)
    delta = np.zeros((M, K))
    for ri, c in enumerate(cats):
        for ci in range(K):
            delta[ri, ci] = ablated[ci].get(c, baseline[c]) - baseline[c]

    # Diagonal-argmin (branch k hurts supercat k most → most negative Δ)
    # Note: ablation HURTS top1 → Δ is negative; argmin = most-negative = most-relied-on.
    diag_hits = 0
    for ri, c in enumerate(cats):
        argmin_branch = int(np.argmin(delta[ri, :]))
        if argmin_branch == c:
            diag_hits += 1
    print()
    print('=' * 80)
    print(f'  Per-supercategory × per-branch top1 drop when branch k zero-ablated')
    print('=' * 80)
    print(f'Diagonal hits (argmin Δ = own assigned branch): {diag_hits}/{M}  '
          f'(random chance: {1/K:.0%}; full specialization: 100%)')
    print(f'Max |Δ| across matrix: {np.max(np.abs(delta)):.2f} top1')
    print(f'Mean |Δ| across matrix: {np.mean(np.abs(delta)):.2f} top1')

    out_json = args.out_json or f'outputs/analysis/vit_diag/{os.path.basename(args.run_dir)}.json'
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        'run_dir': args.run_dir,
        'config_summary': {
            'algorithm': cfg.get('algorithm', {}).get('type'),
            'p_active': cfg.get('algorithm', {}).get('p_active'),
            'shared_expert_dim': cfg.get('model', {}).get('shared_expert_dim', 0),
            'num_branches': K,
        },
        'max_batches': args.max_batches,
        'cats': cats,
        'baseline_top1_per_cat': {int(c): float(baseline[c]) for c in cats},
        'ablated_top1_per_cat': {
            int(k): {int(c): float(ablated[k].get(c, baseline[c])) for c in cats}
            for k in ablated
        },
        'delta_matrix': delta.tolist(),  # M × K
        'diag_hits': diag_hits, 'n_cats': M,
        'max_abs_delta': float(np.max(np.abs(delta))),
        'mean_abs_delta': float(np.mean(np.abs(delta))),
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[Diag-ViT] wrote {out_json}')


if __name__ == '__main__':
    main()
