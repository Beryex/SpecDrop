#!/usr/bin/env python3
"""T3.2 — Inference robustness reframed as upstream-classifier accuracy curve.

The existing E5 noise sweep (`outputs/analysis/noisy_cat_sweep.csv`) reports
top1 vs noise probability p, where p ∈ [0, 1] is the per-sample probability
of replacing the routing cluster_id with a uniform random one. We want
this reframed as "k-way upstream classifier accuracy → ours top1" — direct
response to W5 ("how does method behave when an upstream classifier feeds
noisy cluster_ids?").

Conversion: an upstream classifier with accuracy α produces a noisy
cluster_id with effective replacement probability p = (1 − α) × M/(M − 1)
under the assumption that errors are uniform-random over the M − 1 wrong
classes. Inverting:

    α = 1 − p × (M − 1) / M

For CIFAR M = 20 → α = 1 − 0.95 p.

Outputs:
  outputs/analysis/inference_robustness_curve.json
  outputs/analysis/inference_robustness_curve.md   (markdown table)
  outputs/analysis/inference_robustness_curve.csv  (paper-pgfplots-friendly)

Optional: matplotlib plot at outputs/analysis/inference_robustness_curve.pdf
if --plot is set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--csv_in', default='outputs/analysis/noisy_cat_sweep.csv')
    ap.add_argument('--num_categories', type=int, default=20,
                    help='M for CIFAR superclasses (default 20).')
    ap.add_argument('--out_json', default='outputs/analysis/inference_robustness_curve.json')
    ap.add_argument('--out_md', default='outputs/analysis/inference_robustness_curve.md')
    ap.add_argument('--out_csv', default='outputs/analysis/inference_robustness_curve.csv')
    ap.add_argument('--plot', action='store_true', help='Save PDF figure')
    args = ap.parse_args()

    if not os.path.exists(args.csv_in):
        raise FileNotFoundError(
            f'{args.csv_in} not found. Generate via the E5 noise-sweep ablation '
            f'(scripts/analyze_e5_noise_sweep.py) first.')

    M = args.num_categories
    rows = []
    with open(args.csv_in) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({'method': r['method'], 'seed': int(r['seed']),
                          'noise_p': float(r['noise_p']), 'top1': float(r['top1'])})
    print(f'[robustness] loaded {len(rows)} rows from {args.csv_in}')

    # Convert noise_p → upstream-classifier accuracy α
    # α = 1 - p × (M-1)/M
    factor = (M - 1) / M
    for r in rows:
        r['classifier_acc'] = 1.0 - r['noise_p'] * factor

    # Aggregate: (method, classifier_acc) → list of top1
    grp = defaultdict(list)
    for r in rows:
        key = (r['method'], round(r['classifier_acc'], 6))
        grp[key].append(r['top1'])

    import statistics
    agg = []
    for (method, acc), v in sorted(grp.items()):
        m = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        agg.append({'method': method, 'classifier_acc': acc,
                     'top1_mean': m, 'top1_std': sd, 'n_seeds': len(v),
                     'noise_p': round((1.0 - acc) / factor, 6)})
    methods = sorted({r['method'] for r in agg})
    classifier_accs = sorted({r['classifier_acc'] for r in agg}, reverse=True)

    # Write JSON
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    payload = {
        'num_categories': M,
        'conversion_formula': f'classifier_acc = 1 - noise_p * (M-1)/M  (with M={M})',
        'methods': methods,
        'classifier_acc_grid': classifier_accs,
        'data': agg,
    }
    with open(args.out_json, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[robustness] wrote {args.out_json}')

    # Markdown
    md = ['# Inference Robustness Curve — Upstream Classifier Accuracy → Top-1',
          '',
          f'M = {M} (CIFAR-100 superclasses). Conversion: '
          f'`classifier_acc = 1 - noise_p * {factor:.4f}` (uniform-random misclassification).',
          '',
          '| classifier_acc | noise_p | ' + ' | '.join(methods) + ' |',
          '|---' * (2 + len(methods)) + '|']
    for acc in classifier_accs:
        nb = round((1.0 - acc) / factor, 4)
        cells = [f'{acc:.3f}', f'{nb:.3f}']
        for m in methods:
            v = next((r for r in agg if r['method'] == m and abs(r['classifier_acc'] - acc) < 1e-6), None)
            if v is None:
                cells.append('—')
            elif v['n_seeds'] > 1:
                cells.append(f'{v["top1_mean"]:.2f} ± {v["top1_std"]:.2f}')
            else:
                cells.append(f'{v["top1_mean"]:.2f}')
        md.append('| ' + ' | '.join(cells) + ' |')
    os.makedirs(os.path.dirname(args.out_md) or '.', exist_ok=True)
    with open(args.out_md, 'w') as f:
        f.write('\n'.join(md))
    print(f'[robustness] wrote {args.out_md}')

    # CSV (paper-friendly)
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'classifier_acc', 'noise_p', 'top1_mean', 'top1_std', 'n_seeds'])
        for r in agg:
            w.writerow([r['method'], r['classifier_acc'], r['noise_p'],
                         r['top1_mean'], r['top1_std'], r['n_seeds']])
    print(f'[robustness] wrote {args.out_csv}')

    # Optional plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for m in methods:
                pts = sorted([(r['classifier_acc'], r['top1_mean'], r['top1_std'])
                              for r in agg if r['method'] == m])
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; es = [p[2] for p in pts]
                ax.errorbar(xs, ys, yerr=es, marker='o', label=m, capsize=2)
            ax.set_xlabel(f'Upstream classifier accuracy (M={M}-way, uniform errors)')
            ax.set_ylabel('CIFAR-100 top-1 (%)')
            ax.set_title('Inference robustness vs upstream classifier accuracy')
            ax.invert_xaxis()  # 1.0 (perfect) on left
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf_path = args.out_md.replace('.md', '.pdf')
            fig.savefig(pdf_path)
            print(f'[robustness] wrote {pdf_path}')
        except ImportError:
            print('  WARN: matplotlib not installed; skipping --plot')


if __name__ == '__main__':
    main()
