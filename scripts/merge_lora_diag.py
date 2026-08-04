#!/usr/bin/env python3
"""Merge per-shard LoRA spec diagnostic JSONs into the final aggregated output.

Companion to `diagnose_lora_specialization.py --branch_subset / --skip_baseline`.
When the diagnostic is parallelized across N GPUs, each writes a partial
shard JSON; this script combines them, computes the aggregate stats, and
writes the same JSON schema the non-sharded full run produces.

Usage:
    python scripts/merge_lora_diag.py shard0.json shard1.json shard2.json \\
        --out outputs/analysis/lora_diag/ours_s42.json

Validation:
  - Exactly one shard must include `baseline_per_cat` (the GPU that ran
    the no-ablation baseline). Refuses to merge if zero or >1.
  - Union of `branches_run` must cover 0..K-1 with no duplicates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def merge_shards(shard_paths, out_path):
    shards = []
    for p in shard_paths:
        with open(p) as f:
            shards.append((p, json.load(f)))

    # Validate: all shards from same run + same K
    run_dirs = {s.get('run_dir') for _, s in shards}
    if len(run_dirs) != 1:
        raise ValueError(f'shards from different runs: {run_dirs}')
    Ks = {s.get('K') for _, s in shards}
    if len(Ks) != 1:
        raise ValueError(f'shards report different K values: {Ks}')
    K = Ks.pop()

    # Validate: exactly one shard with baseline
    with_baseline = [(p, s) for p, s in shards if s.get('baseline_per_cat')]
    if len(with_baseline) == 0:
        raise ValueError(
            'No shard has baseline_per_cat — at least one shard must omit '
            '--skip_baseline so the baseline (no-ablation) eval is computed.')
    if len(with_baseline) > 1:
        raise ValueError(
            f'Multiple shards have baseline_per_cat: '
            f'{[p for p, _ in with_baseline]}. Run with --skip_baseline on '
            f'all but one shard to avoid wasted compute.')
    baseline = with_baseline[0][1]['baseline_per_cat']

    # Validate: branches_run union covers 0..K-1, no duplicates
    all_branches = []
    for _, s in shards:
        all_branches.extend(s.get('branches_run', []))
    if sorted(all_branches) != list(range(K)):
        missing = sorted(set(range(K)) - set(all_branches))
        dup = sorted(b for b in set(all_branches) if all_branches.count(b) > 1)
        raise ValueError(
            f'shards do not cleanly cover 0..{K-1}: '
            f'missing={missing}, duplicates={dup}')

    # Combine ablated dicts (shard JSON keys are str-of-int from json.dump)
    ablated = {}
    for _, s in shards:
        for k_str, per_cat in s.get('ablated_per_cat', {}).items():
            k = int(k_str)
            if k in ablated:
                raise ValueError(f'branch {k} present in multiple shards')
            # Inner per-cat keys are str-of-int too — leave for downstream
            ablated[k] = {int(c): v for c, v in per_cat.items()}

    # Baseline keys may be str-of-int (from json) — normalize
    baseline = {int(c): v for c, v in baseline.items()}

    # Compute delta + diag_hits (same logic as full-run path)
    cats = sorted(baseline.keys())
    Mc = len(cats)
    delta = np.zeros((Mc, K))
    for ri, c in enumerate(cats):
        for ci in range(K):
            base_r = baseline[c]['rouge_l']
            abl_for_branch = ablated.get(ci, {})
            abl_r = abl_for_branch.get(c, baseline[c])['rouge_l']
            delta[ri, ci] = abl_r - base_r

    diag_hits = 0
    for ri, c in enumerate(cats):
        argmin_branch = int(np.argmin(delta[ri, :]))
        if argmin_branch == c:
            diag_hits += 1

    print()
    print('=' * 80)
    print(f'  [merge_lora_diag] Combined {len(shards)} shards covering all {K} branches')
    print('=' * 80)
    print(f'Diagonal hits: {diag_hits}/{Mc}  (random: {1/K:.0%}; full spec: 100%)')
    print(f'Max |Δ|: {np.max(np.abs(delta)):.4f} ROUGE-L')
    print(f'Mean |Δ|: {np.mean(np.abs(delta)):.4f}')

    # config_summary: same across shards (assert), take from first
    config_summary = shards[0][1].get('config_summary', {})

    payload = {
        'run_dir': run_dirs.pop(),
        'config_summary': config_summary,
        'cats': cats,
        'baseline_per_cat': {int(c): baseline[c] for c in cats},
        'ablated_per_cat': {
            int(k): {int(c): ablated[k].get(c, baseline[c]) for c in cats}
            for k in ablated
        },
        'delta_matrix': delta.tolist(),
        'diag_hits': diag_hits, 'n_cats': Mc,
        'max_abs_delta': float(np.max(np.abs(delta))),
        'mean_abs_delta': float(np.mean(np.abs(delta))),
        'merged_from_shards': [os.path.basename(p) for p, _ in shards],
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\n[merge_lora_diag] wrote {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('shards', nargs='+', help='Shard JSON paths to merge.')
    ap.add_argument('--out', required=True, help='Final merged JSON path.')
    args = ap.parse_args()
    merge_shards(args.shards, args.out)


if __name__ == '__main__':
    main()
