"""Per-method FLOPs / params audit for the ImageNet ViT main table.

Motivation: K=46 MultiBranch-ViT has
ALL branches active every forward (∫ 1 × K per block), so its FLOPs are
substantially higher than top-k MoE baselines (Mod-Squad top-2, COMET
k-WTA 0.25). Paper Table 2 reports params AND MACs side-by-side so readers
can check "does ours win by compute, not by routing?" directly.

Usage:
    python -m scripts.profile_vit_flops --batch 1 --img-size 224

Depends on fvcore (standard FAIR op-FLOP counter, used by ViT, DINO,
DeiT, Soft MoE releases). Reports per-method MAC count; multiply by 2
for FLOP (FMA convention) when quoting in paper.

Outputs a markdown table to stdout AND a JSON dump for the paper writer
to pick up — see `outputs/analysis/vit_flops.{md,json}`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

# Methods + their config builders. Each returns a (model, name, label) tuple.
# Matches the 7-method main table in rtx5090_6_imagenet_vit_faithful.sh.


def _build(cfg_dict):
    """Build model from minimal config dict (bypasses run.py plumbing)."""
    from models import build_model
    return build_model({'model': cfg_dict, 'algorithm': {'type': 'none'}})


def _method_dense():
    return _build({'type': 'vit_small', 'num_classes': 1000,
                    'img_size': 224, 'patch_size': 16,
                    'drop_path_rate': 0.0}), 'vit_small', 'Dense ViT-S/16'


def _method_mod_squad():
    return _build({'type': 'mod_squad_vit', 'num_classes': 1000,
                    'num_experts': 16, 'top_k': 2, 'mi_weight': 0.001,
                    'drop_path_rate': 0.0}), 'mod_squad_vit', 'Mod-Squad (top-2/16)'


def _method_soft_moe():
    return _build({'type': 'soft_moe_vit', 'num_classes': 1000,
                    'num_experts': 32, 'slots_per_expert': 1,
                    'drop_path_rate': 0.0}), 'soft_moe_vit', 'Soft MoE (N=32)'


def _method_comet():
    return _build({'type': 'comet_vit', 'num_classes': 1000,
                    'p_keep': 0.25, 'drop_path_rate': 0.0}), 'comet_vit', 'COMET (k=0.25)'


def _method_mb_no_routing(se_ratio=0.0):
    from data.slimpajama import split_ffn_budget_for_se
    K = 46
    TOTAL_FFN = 384 * 4
    total_branch_ffn, se_dim = split_ffn_budget_for_se(TOTAL_FFN, se_ratio, K)
    branch_hidden = total_branch_ffn // K
    return _build({'type': 'multi_branch_vit_small', 'num_classes': 1000,
                    'num_branches': K, 'branch_hidden': branch_hidden,
                    'shared_expert_dim': se_dim, 'drop_path_rate': 0.0}), (
        f'mbvit_no_routing_se{se_ratio}' if se_ratio > 0 else 'mbvit_no_routing'), (
        f'MB-ViT no-routing + SE={se_ratio}' if se_ratio > 0 else 'MB-ViT no-routing')


def _method_ours(pa=0.6, beta=4.0, se_ratio=1.0):
    # Ours has the same forward-pass FLOPs as MB-no-routing (same K branches,
    # same shared expert) — the routing mask only reweights outputs, doesn't
    # skip any computation. So FLOPs = MB-ViT + SE.
    from data.slimpajama import split_ffn_budget_for_se
    K = 46
    TOTAL_FFN = 384 * 4
    total_branch_ffn, se_dim = split_ffn_budget_for_se(TOTAL_FFN, se_ratio, K)
    branch_hidden = total_branch_ffn // K
    return _build({'type': 'multi_branch_vit_small', 'num_classes': 1000,
                    'num_branches': K, 'branch_hidden': branch_hidden,
                    'shared_expert_dim': se_dim, 'drop_path_rate': 0.0}), (
        'ours_vit'), (f'Ours (K=46, SE={se_ratio}, pa={pa}, β={beta})')


METHODS = [
    _method_dense,
    _method_mod_squad,
    _method_soft_moe,
    _method_comet,
    _method_mb_no_routing,                  # SE=0
    lambda: _method_mb_no_routing(se_ratio=1.0),   # matched-SE @ nominal SE=1.0
    lambda: _method_ours(pa=0.6, beta=4.0, se_ratio=1.0),
]


def _flops(model, batch_size: int, img_size: int) -> Dict:
    """Return {'macs', 'params', 'trainable'} dict. macs = multiply-accumulate
    count from fvcore; FLOPs = 2 × MACs under standard FMA convention."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print('ERROR: fvcore not installed. Run: pip install fvcore')
        sys.exit(1)

    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size)
    # Some multi-branch models accept an optional branch_mask; pass None to
    # exercise the default (uniform) path which represents the forward compute.
    # FlopCountAnalysis handles the forward for us.
    try:
        flop_counter = FlopCountAnalysis(model, x)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        macs = flop_counter.total()
    except Exception as e:
        # Multi-branch models with required branch_mask arg: feed a dummy mask.
        # mask = ones(B, K) works for uniform-weighted forward.
        K = getattr(model, 'num_branches', None)
        if K is not None:
            mask = torch.ones(batch_size, K)
            # Set a dummy mask_scale so the fixed-denominator merge doesn't raise.
            model.mask_scale = float(K)
            flop_counter = FlopCountAnalysis(model, (x, mask))
            flop_counter.unsupported_ops_warnings(False)
            flop_counter.uncalled_modules_warnings(False)
            macs = flop_counter.total()
        else:
            raise e

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'macs': macs, 'params': params_total, 'trainable': params_trainable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--output-dir', default='./outputs/analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Profiling ViT main-table methods at batch={args.batch}, '
          f'img_size={args.img_size} ...\n')

    rows = []
    for builder in METHODS:
        try:
            model, name, label = builder()
        except Exception as e:
            print(f'  SKIP {builder.__name__}: {e}')
            continue
        stats = _flops(model, args.batch, args.img_size)
        rows.append({
            'name': name,
            'label': label,
            'macs': stats['macs'],
            'gflops': stats['macs'] * 2 / 1e9,   # × 2 for FMA → FLOPs
            'params': stats['params'],
            'trainable': stats['trainable'],
        })
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Markdown table for appendix.
    md_lines = [
        '| Method | Params | Trainable | GFLOPs (@224) |',
        '|---|---|---|---|',
    ]
    for r in rows:
        md_lines.append(
            f"| {r['label']} | {r['params']/1e6:.2f}M "
            f"| {r['trainable']/1e6:.2f}M | {r['gflops']:.2f} |")
    md_text = '\n'.join(md_lines)
    print(md_text)

    with open(os.path.join(args.output_dir, 'vit_flops.md'), 'w') as f:
        f.write(md_text + '\n')
    with open(os.path.join(args.output_dir, 'vit_flops.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'\nWrote {args.output_dir}/vit_flops.{{md,json}}')


if __name__ == '__main__':
    main()
