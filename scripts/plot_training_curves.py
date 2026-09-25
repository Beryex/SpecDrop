#!/usr/bin/env python3
"""App. E.7 Fig. 6: ImageNet BREEDS-46 validation top-1 across training.

Deployed all-blocks Soft MoE, tuned second-half Soft MoE, and Soft SpecDrop (ours):
3-seed mean per epoch (line) and seed min/max (band), read from the per-epoch
`epoch_history[*].test_top1` of each run's results.json ("test" there is the
ImageNet-1K validation split, the evaluation split of every ImageNet table).

Usage (from the repo root):
  python scripts/plot_training_curves.py --output outputs/analysis/fig_training_curves.pdf
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

RUNS = [('Soft MoE, deployed (all blocks)', 'outputs/rtx5090_imagenet_vit_faithful/soft_moe_vit_s%d', '#009E73'),
        ('Soft MoE, tuned (2nd-half)',      'outputs/softmoe_study_finals/softmoe_study_r7_s%d',  '#D55E00'),
        ('Soft SpecDrop (ours)',            'outputs/rtx5090_imagenet_vit_faithful/ours_vit_s%d',  '#0072B2')]
SEEDS = (42, 123, 456)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', default='outputs/analysis/fig_training_curves.pdf')
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(424.420625 / 72, 251.18375 / 72))
    for label, pattern, color in RUNS:
        hist = np.array([[e['test_top1'] for e in
                          json.load(open(pattern % s + '/results.json'))['epoch_history']]
                         for s in SEEDS])
        x = np.arange(1, hist.shape[1] + 1)
        mean = hist.mean(0)
        ax.fill_between(x, hist.min(0), hist.max(0), color=color, alpha=0.15, lw=0)
        ax.plot(x, mean, color=color, lw=1.6, label=label)
        ax.text(x[-1] + 1.5, mean[-1], f'{mean[-1]:.1f}', color=color, fontsize=8, va='center')
    ax.set_xlim(1, 108)
    ax.set_ylim(0, 85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ImageNet BREEDS-46 validation top-1 (%)')
    ax.grid(axis='y', color='0.9', lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == '__main__':
    main()
