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


METHOD_TITLES = {'ours': 'Soft SpecDrop (ours)', 'no_routing': 'No-Routing'}


def diag_stats(kd_matrix):
    """Per-category diag-argmax hits (ties -> lowest branch index, as in the
    Align column) and mean(|diag|) / mean(|off-diag|) on the raw |Delta|."""
    a = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    M, K = a.shape
    n = min(M, K)
    hits = sum(int(a[c].argmax() == c) for c in range(n))
    diag = np.mean([a[c, c] for c in range(n)])
    off = (a.sum() - sum(a[c, c] for c in range(n))) / (a.size - n)
    return hits, n, diag / max(off, 1e-12)


def plot_grid(method_kds, out_pdf):
    """Side-by-side heatmaps of |Delta_{k,c}|, each row normalized to its max.

    Drawn at its printed size (full text width, 5.5 in) so font sizes are
    true points: square cells, superclass names on the shared y-axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    methods = list(method_kds.keys())
    n = len(methods)
    M, K = next(iter(method_kds.values())).shape

    # Row-normalize |Delta| so each superclass shows which branch matters most
    # for it (the per-category argmax that Align counts).
    normed = {}
    for m, kd in method_kds.items():
        a = np.abs(kd)
        normed[m] = a / np.maximum(a.max(axis=1, keepdims=True), 1e-12)

    fig, axes = plt.subplots(1, n, figsize=(5.5, 3.2), squeeze=False,
                             gridspec_kw={'wspace': 0.05, 'left': 0.285,
                                          'right': 0.915, 'top': 0.87,
                                          'bottom': 0.12})
    axes = axes[0]
    cat_names = [c.replace('_', ' ') for c in SUPERCLASS_NAMES[:M]]
    for ax, method in zip(axes, methods):
        im = ax.imshow(normed[method], aspect='equal', cmap='OrRd',
                       interpolation='nearest', vmin=0, vmax=1)
        hits, n_diag, ratio = diag_stats(method_kds[method])
        ax.set_title(METHOD_TITLES.get(method, method), fontsize=8, pad=11)
        ax.text(0.5, 1.012, f'diag-argmax {hits}/{n_diag}, ratio {ratio:.2f}$\\times$',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5,
                color='#333333')
        ax.set_xticks(range(0, K, 5))
        ax.set_xticks(range(K), minor=True)
        ax.tick_params(axis='x', labelsize=7, length=2, pad=1.5)
        ax.tick_params(which='minor', length=1)
        ax.set_xlabel('Branch $k$', fontsize=7.5, labelpad=1.5)
        if ax is axes[0]:
            ax.set_yticks(range(M))
            ax.set_yticklabels(cat_names, fontsize=6.0)
            ax.tick_params(axis='y', length=2, pad=1.5)
        else:
            ax.set_yticks(range(M))
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=2)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color('#666666')
        # Outline each superclass's round-robin assigned branch (ours only).
        if method == 'ours':
            for c in range(M):
                ax.add_patch(plt.Rectangle((c % K - 0.5, c - 0.5), 1, 1,
                                           fill=False, edgecolor='#1f4e9c',
                                           linewidth=0.9))

    fig.canvas.draw()  # resolve equal-aspect axes positions before placing the colorbar
    pos = axes[-1].get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.013, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label('$|\\Delta_{k,c}|$ / row max', fontsize=7, labelpad=3)

    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f'  saved {out_pdf}')
    for m in methods:
        hits, n_diag, ratio = diag_stats(method_kds[m])
        print(f'  {m}: diag-argmax {hits}/{n_diag}, diag/off {ratio:.4f}')


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
