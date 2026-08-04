#!/usr/bin/env python3
"""E3 — Aggregate denom-ablation runs into one CSV.

Walks `outputs/rtx5090_denom_ablation/{config}_s{seed}/results.json` for the
listed configs (default: stoch_fixed, stoch_naive, rand_fixed, rand_naive)
and the existing `outputs/rtx5090_cifar100_faithful/ours_s{seed}/results.json`
as the soft_fixed reference, then writes:

    outputs/rtx5090_denom_ablation/denom_ablation_summary.csv
    columns: config, mean_top1, std, gap_vs_soft_fixed
"""

import argparse
import csv
import json
import os
import statistics
import sys


DEFAULT_CONFIGS = ['stoch_fixed', 'stoch_naive', 'rand_fixed', 'rand_naive']
DEFAULT_SEEDS = [42, 123, 456]


def load_top1(run_dir):
    p = os.path.join(run_dir, 'results.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get('best_top1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_dir', default='outputs/rtx5090_denom_ablation')
    parser.add_argument('--ours_dir', default='outputs/rtx5090_cifar100_faithful')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--out_csv',
                        default='outputs/rtx5090_denom_ablation/denom_ablation_summary.csv')
    args = parser.parse_args()

    # soft_fixed reference (= ours)
    soft_fixed_top1s = []
    for seed in args.seeds:
        v = load_top1(os.path.join(args.ours_dir, f'ours_s{seed}'))
        if v is not None:
            soft_fixed_top1s.append(v)
    soft_fixed_mean = statistics.mean(soft_fixed_top1s) if soft_fixed_top1s else None

    rows = []
    if soft_fixed_top1s:
        rows.append({
            'config': 'soft_fixed (ours, ref)',
            'n_seeds': len(soft_fixed_top1s),
            'mean_top1': round(soft_fixed_mean, 2),
            'std': round(statistics.stdev(soft_fixed_top1s), 3) if len(soft_fixed_top1s) > 1 else 0.0,
            'gap_vs_soft_fixed': 0.0,
            'per_seed': ', '.join(f'{v:.2f}' for v in soft_fixed_top1s),
        })

    for cfg in args.configs:
        per_seed = []
        for seed in args.seeds:
            v = load_top1(os.path.join(args.ablation_dir, f'{cfg}_s{seed}'))
            if v is None and seed == 42:
                # configs/cv/{cfg}.yaml ships with seed 42 and an un-suffixed
                # output_dir; accept that layout as the s42 run.
                v = load_top1(os.path.join(args.ablation_dir, cfg))
            if v is not None:
                per_seed.append(v)
        if not per_seed:
            rows.append({'config': cfg, 'n_seeds': 0, 'mean_top1': '', 'std': '',
                         'gap_vs_soft_fixed': '', 'per_seed': '(no runs found)'})
            continue
        mean = statistics.mean(per_seed)
        std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        gap = (mean - soft_fixed_mean) if soft_fixed_mean is not None else None
        rows.append({
            'config': cfg,
            'n_seeds': len(per_seed),
            'mean_top1': round(mean, 2),
            'std': round(std, 3),
            'gap_vs_soft_fixed': round(gap, 2) if gap is not None else '',
            'per_seed': ', '.join(f'{v:.2f}' for v in per_seed),
        })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = ['config', 'n_seeds', 'mean_top1', 'std',
                  'gap_vs_soft_fixed', 'per_seed']
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[E3] wrote {args.out_csv}')
    for r in rows:
        print(f'  {r["config"]:>30}: mean={r["mean_top1"]} ±{r["std"]} '
              f'(gap={r["gap_vs_soft_fixed"]}) seeds={r["per_seed"]}')


if __name__ == '__main__':
    main()
