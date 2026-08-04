#!/usr/bin/env python3
"""E2 — Pruning sensitivity heatmap comparison.

For each method, run pruning sensitivity analysis on seed=42 (already cached
by analyze_e1_mi_table.py if available) and produce:

  - outputs/analysis/heatmap_{method}_s42.csv  : 20×20 raw KD matrix
  - outputs/analysis/fig_heatmap_comparison.pdf : 5-subplot side-by-side

Reuses `evaluate_specialization` (which calls `_compute_pruning_sensitivity`).
If a per-method JSON already exists in outputs/analysis/specialization/, the
KD matrix is loaded from there (no re-inference), otherwise inference is run.

Methods: ours, no_routing (others without category-conditioned branches are
skipped — paper plots ours vs no_routing as the central contrast; if archive
checkpoints for soft_moe/mod_squad/hash_routing become available later they
can be added via --methods).
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from data import get_dataloaders
from data.cifar100 import SUPERCLASS_NAMES
from models import build_model
from algorithms import build_algorithm
from evaluation.metrics import _compute_pruning_sensitivity


DEFAULT_METHODS = ['ours', 'no_routing']  # methods with branch_mask routing on disk


def _load_results(run_dir):
    p = os.path.join(run_dir, 'results.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _load_or_compute_kd(method, seed, base_dir, device, batch_size):
    """Return (kd_matrix, param_counts) for the given (method, seed)."""
    cache_path = os.path.join('outputs', 'analysis', 'specialization',
                              f'{method}_s{seed}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            spec = json.load(f)
        kd = spec.get('pruning_sensitivity')
        if kd is not None:
            return np.array(kd['kd_matrix']), kd['branch_param_counts']

    run_dir = os.path.join(base_dir, f'{method}_s{seed}')
    results = _load_results(run_dir)
    if results is None:
        return None, None
    cfg = results['config']

    ckpt_path = os.path.join(run_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(cfg).to(device)
    algorithm = build_algorithm(cfg)
    # Checkpoints don't save algorithm runtime state. Advance both
    # current_epoch and current_step so progress=1.0 under any warmup_unit.
    # See scripts/_diag_helpers.py for the rationale (step-mode bug surfaced
    # 2026-05-02).
    from scripts._diag_helpers import advance_softspecdrop_to_terminal
    advance_softspecdrop_to_terminal(algorithm,
                                       cfg.get('training', {}).get('epochs', 0))
    if (algorithm is not None and hasattr(model, 'mask_scale')
            and algorithm.expected_mask_sum is not None):
        model.mask_scale = algorithm.expected_mask_sum
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()

    dcfg = cfg.get('data', {})
    _, test_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=batch_size,
        num_workers=dcfg.get('num_workers', 0),
        device=device,
    )

    kd = _compute_pruning_sensitivity(model, test_loader, algorithm, device)
    if kd is None:
        return None, None
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Save just the pruning sensitivity slice (mirroring evaluate_specialization output).
    with open(cache_path, 'w') as f:
        json.dump({'pruning_sensitivity': kd}, f)
    return np.array(kd['kd_matrix']), kd['branch_param_counts']


def save_heatmap_csv(method, seed, kd_matrix, out_dir):
    M, K = kd_matrix.shape
    out_csv = os.path.join(out_dir, f'heatmap_{method}_s{seed}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['superclass'] + [f'branch_{k}' for k in range(K)])
        for c in range(M):
            cat_name = SUPERCLASS_NAMES[c] if c < len(SUPERCLASS_NAMES) else f'cat_{c}'
            w.writerow([cat_name] + [f'{kd_matrix[c, k]:.6e}' for k in range(K)])
    print(f'  saved {out_csv}')


def plot_grid(method_kds, out_pdf):
    """Side-by-side heatmaps with a shared color scale (per-row normalized)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = list(method_kds.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1, 7), squeeze=False)
    axes = axes[0]

    # Normalize each method's KD per-category for visual contrast.
    normed = {}
    for m, kd in method_kds.items():
        row_max = kd.max(axis=1, keepdims=True)
        normed[m] = kd / np.maximum(row_max, 1e-12)

    M, K = next(iter(method_kds.values())).shape
    for ax, method in zip(axes, methods):
        im = ax.imshow(normed[method], aspect='auto',
                        cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(method, fontsize=11)
        ax.set_xlabel('Branch')
        if ax is axes[0]:
            cat_names = SUPERCLASS_NAMES[:M]
            ax.set_yticks(range(M))
            ax.set_yticklabels([n.replace('_', ' ') for n in cat_names], fontsize=7)
            ax.set_ylabel('Superclass')
        else:
            ax.set_yticks([])
        ax.set_xticks(range(0, K, max(1, K // 10)))
        # Mark assignment-matrix diagonal (round-robin for ours).
        if method == 'ours':
            for c in range(M):
                assigned = c % K
                ax.add_patch(plt.Rectangle((assigned - 0.5, c - 0.5), 1, 1,
                                            fill=False, edgecolor='blue',
                                            linewidth=1.2))

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02,
                 label='Per-row-normalized pruning sensitivity')
    plt.suptitle('Pruning sensitivity KD(k, c) heatmap — only category-conditioned '
                 'routing produces a clean diagonal',
                 fontsize=12, y=1.02)
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_pdf}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='outputs/rtx5090_cifar100_faithful')
    parser.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='outputs/analysis')
    parser.add_argument('--out_pdf', default='outputs/analysis/fig_heatmap_comparison.pdf')
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

    os.makedirs(args.out_dir, exist_ok=True)
    method_kds = {}
    for method in args.methods:
        print(f'[E2] {method}_s{args.seed}', flush=True)
        kd, _ = _load_or_compute_kd(method, args.seed, args.base_dir, device,
                                     args.batch_size)
        if kd is None:
            print(f'  SKIP (no checkpoint)')
            continue
        save_heatmap_csv(method, args.seed, kd, args.out_dir)
        method_kds[method] = kd

    if method_kds:
        plot_grid(method_kds, args.out_pdf)
    else:
        print('[E2] no methods produced a KD matrix; nothing to plot.')


if __name__ == '__main__':
    main()
