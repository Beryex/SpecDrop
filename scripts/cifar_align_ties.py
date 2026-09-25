#!/usr/bin/env python3
"""App. E.1 "Ties in CIFAR-100 Align": tie counts and tie-handling variants.

Reads the CIFAR pruning-sensitivity JSONs written by
scripts/diagnose_cifar_specialization.py (scripts/experiments/alignment/run.sh)
and reports, per seed:
  - superclasses whose assigned branch, when zero-ablated, drives their
    accuracy to exactly zero, and off-diagonal (superclass, branch) pairs
    that do the same;
  - superclasses whose most-sensitive branch is tied;
  - the Align fraction under four tie rules: lowest branch index (the rule of
    Tab. 1 and scripts/aggregate_alignment.py), tied credit split equally,
    unique maxima only, and any tied maximum.
Across seeds: mean +/- sample std (N-1), as in the paper.

Usage (from the repo root):
  python scripts/cifar_align_ties.py [--method ours] [--seeds 42 123 456]
"""
import argparse
import json
import os
import statistics

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--diag_dir', default='outputs/analysis/specialization')
    ap.add_argument('--method', default='ours')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()

    rules = {'lowest index': [], 'split credit': [], 'unique maxima': [],
             'any tied maximum': []}
    for seed in args.seeds:
        path = os.path.join(args.diag_dir, f'{args.method}_s{seed}.json')
        with open(path) as f:
            ps = json.load(f)['pruning_sensitivity']
        # kd[c, k] = (full_acc[c] - ablated_acc[c]) / params[k]
        # (evaluation.metrics); rows = superclasses, cols = branches.
        kd = np.abs(np.asarray(ps['kd_matrix'], dtype=np.float64))
        drop = kd * np.asarray(ps['branch_param_counts'], dtype=np.float64)[None, :]
        full = np.asarray(ps['full_acc_per_category'], dtype=np.float64)
        zero = np.isclose(drop, full[:, None], rtol=0, atol=1e-9)
        n = min(kd.shape)
        hits = {r: 0.0 for r in rules}
        tied = 0
        for c in range(n):
            top = np.flatnonzero(kd[c] == kd[c].max())
            tied += len(top) > 1
            hits['lowest index'] += top[0] == c
            hits['split credit'] += (c in top) / len(top)
            hits['unique maxima'] += len(top) == 1 and top[0] == c
            hits['any tied maximum'] += c in top
        diag_zero = int(sum(zero[c, c] for c in range(n)))
        print(f'seed {seed}: assigned branch -> zero accuracy for {diag_zero}/{n} '
              f'superclasses, off-diagonal zero pairs {int(zero.sum()) - diag_zero}, '
              f'tied most-sensitive branch for {tied}/{n}')
        for r in rules:
            rules[r].append(100.0 * hits[r] / n)
        print('    ' + ', '.join(f'{r} {hits[r]:g}/{n}' for r in rules))
    for r, v in rules.items():
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f'{r:17s}: {statistics.mean(v):.1f} +/- {sd:.1f}')


if __name__ == '__main__':
    main()
