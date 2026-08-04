#!/usr/bin/env python3
"""5-panel specialization heatmap: compare Phase D / P / W (pa=0.6, 0.7) / Z.

For each trained checkpoint, the Δ matrix (rows = categories, cols = branches,
values = PPL increase when that branch is zero-ablated) is plotted as a heatmap
with the diagonal highlighted. The 5-panel grid lets a reader see at a glance:
  - whether specialization is present (diagonal brighter than off-diagonal)
  - how specialization strength changes across configs (Phase D vs W vs Z)
  - whether cluster-labeled runs form as clean a diagonal as domain-labeled

Paper appendix figure. Produces both PNG (for Notion/Slack) and PDF (for LaTeX).

Usage:
    python scripts/plot_specialization_heatmaps.py \\
        --diag-dir outputs/analysis/nlp_diag \\
        --output outputs/analysis/specialization_heatmaps

Runs locally — only needs the 5 diag JSON files rsync'd back from 5090.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Checkpoint order + display titles
RUNS = OrderedDict([
    ('phaseD_pa0.6_pi0.4_s42',         'Phase D  pa=0.6\n(scalar, domain)'),
    ('phaseP_pa0.6_wr1.0_s42',         'Phase P  pa=0.6\n(per-cat, domain)'),
    ('phaseW_cluster_pa0.6_wr1.0_s42', 'Phase W  pa=0.6\n(per-cat, cluster k=7)'),
    ('phaseW_cluster_pa0.7_wr1.0_s42', 'Phase W  pa=0.7\n(per-cat, cluster k=7)'),
    ('phaseZ_cluster_k3_pa0.6_wr1.0_s42', 'Phase Z  pa=0.6\n(per-cat, cluster k=3)'),
])

DOMAIN_SHORT = {
    'RedPajamaCommonCrawl': 'CC',
    'RedPajamaC4': 'C4',
    'RedPajamaGithub': 'Github',
    'RedPajamaBook': 'Book',
    'RedPajamaArXiv': 'ArXiv',
    'RedPajamaWikipedia': 'Wiki',
    'RedPajamaStackExchange': 'SE',
}


def load_delta_matrix(diag_json):
    d = json.load(open(diag_json))
    K = d['config_summary']['num_branches']
    using_clusters = d.get('using_clusters', False)
    baseline = {int(k): v for k, v in d['baseline_ppl_per_domain'].items()}
    ablated = {int(k): {int(kk): vv for kk, vv in v.items()}
               for k, v in d['ablated_ppl_per_domain'].items()}
    cats = sorted(baseline.keys())

    delta = np.full((len(cats), K), np.nan)
    for i, c in enumerate(cats):
        for k in range(K):
            if k in ablated and c in ablated[k]:
                delta[i, k] = ablated[k][c] - baseline[c]

    id2n = {int(k): v for k, v in d.get('category_id_to_name', {}).items()}
    if not id2n and using_clusters:
        id2n = {c: f'C{c}' for c in cats}
    elif not id2n:
        id2n = {int(v): k for k, v in d.get('domain_id_to_name', {}).items()}

    # Compact row labels
    row_labels = [DOMAIN_SHORT.get(id2n.get(c, ''), id2n.get(c, f'c{c}'))
                   for c in cats]
    col_labels = [f'b{k}' for k in range(K)]
    diag_hits = d.get('diag_hits')
    n_cats = d.get('n_domains') or len(cats)
    max_abs = d.get('max_abs_delta')

    return delta, row_labels, col_labels, diag_hits, n_cats, max_abs


def plot_one(ax, delta, row_labels, col_labels, title, diag_hits, n_cats, max_abs,
              vmax=None):
    """Plot a single heatmap on the given axes."""
    if vmax is None:
        vmax = np.nanmax(np.abs(delta))
    im = ax.imshow(delta, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)

    # Tick labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('branch ablated', fontsize=9)
    ax.set_ylabel('category', fontsize=9)

    # Cell annotations (the Δ values)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta[i, j]
            if np.isnan(v):
                continue
            # Readable text color: white on dark, black on bright
            color = 'white' if v < vmax * 0.55 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7, color=color)

    # Highlight diagonal (only for square matrices; rows = cats, cols = branches
    # with round_robin assignment makes category c ↔ branch c).
    for i in range(min(delta.shape[0], delta.shape[1])):
        rect = mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor='red', linewidth=1.8)
        ax.add_patch(rect)

    title_full = f'{title}\ndiag {diag_hits}/{n_cats}, max|Δ|={max_abs:.1f}'
    ax.set_title(title_full, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag-dir', default='outputs/analysis/nlp_diag')
    ap.add_argument('--output', default='outputs/analysis/specialization_heatmaps')
    args = ap.parse_args()

    mats = []
    for run_key, title in RUNS.items():
        path = os.path.join(args.diag_dir, f'{run_key}.json')
        if not os.path.exists(path):
            print(f'[plot] skip {run_key} (missing)', file=sys.stderr)
            continue
        mats.append((run_key, title, *load_delta_matrix(path)))

    if not mats:
        print('no diag JSONs found', file=sys.stderr); sys.exit(1)

    # Per-panel vmax so each heatmap uses its own dynamic range. Alternative
    # would be a shared vmax for absolute comparability; we prefer per-panel
    # since Phase Z's max|Δ| (34) would crush the detail in Phase D (21).
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    ims = []
    for ax, (run_key, title, delta, rows, cols, diag, n_cats, mx) in zip(axes, mats):
        im = plot_one(ax, delta, rows, cols, title, diag, n_cats, mx)
        ims.append(im)

    # Each panel has its own colorbar
    for ax, im in zip(axes, ims):
        plt.colorbar(im, ax=ax, shrink=0.7, label='Δ PPL')

    fig.suptitle('Per-category × per-branch ablation Δ PPL '
                  '(red box = round_robin diagonal)', fontsize=11)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for ext in ('png', 'pdf'):
        path = f'{args.output}.{ext}'
        fig.savefig(path, dpi=(200 if ext == 'png' else None), bbox_inches='tight')
        print(f'[plot] saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
