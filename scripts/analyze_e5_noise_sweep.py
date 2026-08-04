#!/usr/bin/env python3
"""E5 — Noisy category-label robustness sweep at inference.

For ours_s{42, 123, 456}, evaluate top-1 with the routing label corrupted
to a uniform random superclass with probability p ∈ {0, 0.05, 0.10, 0.20,
0.50, 1.00}. Each (seed, p) cell is one full val-set forward.

Outputs:
  - outputs/analysis/noisy_cat_sweep.csv  (seed, noise_p, top1)
  - outputs/analysis/fig_noise_robustness.pdf  (line plot, error bars over seeds)

This addresses paper §5 Limitations: at inference the category predictor may
be imperfect; we want to characterize the gentle-degradation regime.
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
from models import build_model
from algorithms import build_algorithm
from evaluation.metrics import evaluate_accuracy


DEFAULT_PS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
DEFAULT_SEEDS = [42, 123, 456]


def _load_results(run_dir):
    p = os.path.join(run_dir, 'results.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def evaluate_at_p(method, seed, p, base_dir, device, batch_size, noise_seed):
    run_dir = os.path.join(base_dir, f'{method}_s{seed}')
    results = _load_results(run_dir)
    if results is None:
        return None
    cfg = results['config']
    ckpt_path = os.path.join(run_dir, 'best.pt')
    if not os.path.exists(ckpt_path):
        return None

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd)

    dcfg = cfg.get('data', {})
    _, test_loader = get_dataloaders(
        data_dir=dcfg.get('data_dir', './data_cache'),
        batch_size=batch_size,
        num_workers=dcfg.get('num_workers', 0),
        device=device,
    )
    # Reproducible noise across runs of the same (seed, p).
    torch.manual_seed(noise_seed)
    metrics = evaluate_accuracy(model, test_loader, algorithm, device,
                                 noise_probability=p)
    return metrics['top1']


def plot_sweep(rows, out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    by_p = {}
    for r in rows:
        by_p.setdefault(r['noise_p'], []).append(r['top1'])
    ps = sorted(by_p.keys())
    means = np.array([np.mean(by_p[p]) for p in ps])
    stds = np.array([np.std(by_p[p]) for p in ps])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(ps, means, yerr=stds, marker='o', capsize=4, color='#0072B2',
                 label='Soft SpecDrop (ours), 3 seeds')
    ax.set_xlabel('Noise probability p (per-sample category-label corruption)')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_title('E5 — Noisy category label robustness at inference')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[E5] saved {out_pdf}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='outputs/rtx5090_cifar100_faithful')
    parser.add_argument('--method', default='ours')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--ps', nargs='+', type=float, default=DEFAULT_PS)
    parser.add_argument('--out_csv', default='outputs/analysis/noisy_cat_sweep.csv')
    parser.add_argument('--out_pdf', default='outputs/analysis/fig_noise_robustness.pdf')
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--noise_seed', type=int, default=2026)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[E5] device={device}, method={args.method}, seeds={args.seeds}, ps={args.ps}')

    rows = []
    for seed in args.seeds:
        for p in args.ps:
            print(f'  → seed={seed}, p={p}', flush=True)
            top1 = evaluate_at_p(args.method, seed, p, args.base_dir, device,
                                  args.batch_size, args.noise_seed)
            if top1 is None:
                print(f'     SKIP (no checkpoint)')
                continue
            rows.append({'method': args.method, 'seed': seed,
                         'noise_p': p, 'top1': round(top1, 4)})
            print(f'     top1={top1:.2f}%')

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'seed', 'noise_p', 'top1'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[E5] wrote {args.out_csv}')

    if rows:
        plot_sweep(rows, args.out_pdf)


if __name__ == '__main__':
    main()
