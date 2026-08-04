#!/usr/bin/env python3
"""E1 — MI + alignment table across all CIFAR-100 baselines.

Loads each `outputs/rtx5090_cifar100_faithful/{method}_s{seed}/best.pt`, runs
val-set forward, and writes `outputs/analysis/mi_table.csv` with one row per
(method, seed):

    method, seed, mi, alignment, usage_entropy, unique_branches

Methods that do NOT have a category-conditioned multi-branch mechanism
(dense ResNets, ETD, ContextualDropout) are skipped — their lines in the
CSV are written as `n/a` so the paper table can show them as "—".
"""

import argparse
import csv
import json
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data import get_dataloaders
from models import build_model
from algorithms import build_algorithm
from evaluation.metrics import evaluate_specialization


DEFAULT_METHODS = ['ours', 'no_routing', 'stoch_depth', 'et_dropout',
                   'ctx_dropout', 'resnet110']
DEFAULT_SEEDS = [42, 123, 456]


def _load_results(run_dir):
    p = os.path.join(run_dir, 'results.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _load_best_state(run_dir, device):
    p = os.path.join(run_dir, 'best.pt')
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location=device, weights_only=False)


def _has_multi_branch_routing(cfg):
    """True if this method has branch_mask routing we can compute MI/alignment for."""
    model_type = cfg.get('model', {}).get('type', '')
    algo_type = cfg.get('algorithm', {}).get('type', 'none')
    if model_type not in ('multi_branch_resnet110', 'multi_branch_resnet110_forloop',
                          'grouped_multi_branch_resnet110', 'multi_branch'):
        return False
    return algo_type != 'none'


def analyze_one(method, seed, base_dir, device, batch_size=128):
    run_dir = os.path.join(base_dir, f'{method}_s{seed}')
    if not os.path.isdir(run_dir):
        return {'method': method, 'seed': seed, 'status': 'missing',
                'mi': '', 'alignment': '', 'usage_entropy': '',
                'unique_branches': '', 'best_top1': ''}

    results = _load_results(run_dir)
    if results is None:
        return {'method': method, 'seed': seed, 'status': 'no_results_json',
                'mi': '', 'alignment': '', 'usage_entropy': '',
                'unique_branches': '', 'best_top1': ''}
    cfg = results.get('config', {})
    best_top1 = results.get('best_top1', '')

    if not _has_multi_branch_routing(cfg):
        return {'method': method, 'seed': seed, 'status': 'n/a (no branch routing)',
                'mi': '', 'alignment': '', 'usage_entropy': '',
                'unique_branches': '', 'best_top1': best_top1}

    ckpt = _load_best_state(run_dir, device)
    if ckpt is None:
        return {'method': method, 'seed': seed, 'status': 'no_best_pt',
                'mi': '', 'alignment': '', 'usage_entropy': '',
                'unique_branches': '', 'best_top1': best_top1}

    # Build model + algorithm matching the saved config
    model = build_model(cfg).to(device)
    algorithm = build_algorithm(cfg)
    # Checkpoints don't save algorithm runtime state. For SoftSpecDrop with
    # warmup, a freshly built algorithm sits at progress=0 (uniform weights
    # ≡ no_routing) — under either warmup_unit='epoch' (current_epoch=0) or
    # 'step' (current_step=0). Advance BOTH so post-fix runs work for any
    # mode. See scripts/_diag_helpers.py for the full rationale.
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    advance_softspecdrop_to_terminal(algorithm,
                                       cfg.get('training', {}).get('epochs', 0))
    if algorithm is not None:
        if hasattr(model, 'mask_scale') and algorithm.expected_mask_sum is not None:
            model.mask_scale = algorithm.expected_mask_sum

    state_dict = ckpt['model_state_dict']
    # torch.compile saves with `_orig_mod.` prefix; strip it.
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    dcfg = cfg.get('data', {})
    _, test_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=batch_size,
        num_workers=dcfg.get('num_workers', 0),
        device=device,
    )

    spec = evaluate_specialization(model, test_loader, algorithm, device)
    if spec is None:
        return {'method': method, 'seed': seed, 'status': 'spec_returned_none',
                'mi': '', 'alignment': '', 'usage_entropy': '',
                'unique_branches': '', 'best_top1': best_top1}

    # Persist per-(method, seed) JSON for downstream re-analysis (E2 reuses kd_matrix).
    out_dir = os.path.join('outputs', 'analysis', 'specialization')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{method}_s{seed}.json'), 'w') as f:
        json.dump(spec, f, indent=2)

    return {
        'method': method,
        'seed': seed,
        'status': 'ok',
        'mi': round(spec['mutual_information'], 4),
        'alignment': round(spec['alignment'], 4),
        'usage_entropy': round(spec['usage_entropy'], 4),
        'unique_branches': spec['unique_branches'],
        'best_top1': best_top1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='outputs/rtx5090_cifar100_faithful')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--out_csv', default='outputs/analysis/mi_table.csv')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[E1] device={device}, base_dir={args.base_dir}')

    rows = []
    for method in args.methods:
        for seed in args.seeds:
            print(f'  → {method}_s{seed}', flush=True)
            row = analyze_one(method, seed, args.base_dir, device, args.batch_size)
            rows.append(row)
            if row['status'] == 'ok':
                print(f'     MI={row["mi"]:.4f}  align={row["alignment"]:.4f}  '
                      f'H={row["usage_entropy"]:.4f}  uniq={row["unique_branches"]}  '
                      f'top1={row["best_top1"]}')
            else:
                print(f'     {row["status"]}')

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = ['method', 'seed', 'status', 'mi', 'alignment',
                  'usage_entropy', 'unique_branches', 'best_top1']
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[E1] wrote {args.out_csv}')


if __name__ == '__main__':
    main()
