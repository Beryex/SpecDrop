#!/usr/bin/env python3
"""T2.2 — Fisher exact / chi-square test on per-task ROUGE Δ × Wang 2022 cluster_id.

For each of the 119 SuperNI held-out tasks at s42, compute Δ = ours − HydraLoRA,
classify each task as ours-wins / Hydra-wins / tied (within ε), then test for
association with the task's Wang 2022 cluster_id.

Motivating check: if crossing the per-task win pattern with the Wang 2022
cluster IDs yields Fisher-exact p<0.05, the granularity-alignment evidence
scales from n=4 settings to n=119 tasks. Goal: show that the ours-wins
distribution across clusters is non-uniform (clusters where ours dominates
exist), supporting the granularity-alignment thesis at fine task grain.

Inputs:
  outputs/eval_lora_per_task/{ours,hydra_lora_n8}_s42.json   (B1 output)
  data_cache/lora/superni_task_to_cluster.json               (built by the LoRA data loader)

If the cluster mapping file is missing, exits with a clear error + rsync
command suggestion. The mapping is small (~10KB) and deterministic, built
once by data.superni_domain_map.build_or_load_domain_map.

Output:
  outputs/analysis/lora_per_task_fisher.json
    {
      'n_tasks': 119,
      'mean_delta': float,
      'wins_ours': int, 'wins_hydra': int, 'ties': int,
      'cluster_breakdown': {cluster_id: {ours, hydra, tie, n}},
      'tests': {
          'chi2_2way': {'stat': ..., 'p': ...},   # ours-wins vs others by cluster
          'chi2_3way': {'stat': ..., 'p': ...},   # ours/hydra/tie by cluster
          'fisher_per_cluster': {cluster_id: {'odds_ratio': ..., 'p': ...}},
      },
      'top_clusters_for_ours': [(cluster_id, win_rate, n_tasks), ...],
      'top_clusters_for_hydra': [(cluster_id, win_rate, n_tasks), ...],
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load_or_die(path, hint=''):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Required file not found: {path}\n'
            f'{hint}'.strip())
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ours', default='outputs/eval_lora_per_task/ours_s42.json')
    ap.add_argument('--hydra', default='outputs/eval_lora_per_task/hydra_lora_n8_s42.json')
    ap.add_argument('--mapping', default='data_cache/lora/superni_task_to_cluster.json',
                    help='task_id → cluster_id mapping JSON. Build via '
                         'data.superni_domain_map.build_or_load_domain_map (auto-generated '
                         'on first LoRA training run; cached under data_cache/lora/).')
    ap.add_argument('--tie_eps', type=float, default=0.0,
                    help='|Δ| ≤ tie_eps counts as a tie (default 0 = strict).')
    ap.add_argument('--out', default='outputs/analysis/lora_per_task_fisher.json')
    args = ap.parse_args()

    # 1. Load per-task ROUGE for the two methods
    ours = _load_or_die(args.ours,
        'Run B1 first (rtx5090_11 F per-task) on the s42 ours checkpoint.')
    hydra = _load_or_die(args.hydra,
        'Run B1 first on s42 hydra_lora_n8 checkpoint.')

    # 2. Load task → cluster_id mapping
    mapping_raw = _load_or_die(args.mapping,
        'Build via data.superni_domain_map.build_or_load_domain_map on the 5090, '
        '— run any LoRA chain once to generate it (small file, ~10KB).')
    # Tolerate the two shapes: (a) flat {tid: cid} dict, (b) full domain_map
    # dict with task_to_cluster nested inside.
    if 'task_to_cluster' in mapping_raw:
        task_to_cluster = mapping_raw['task_to_cluster']
        K = mapping_raw.get('K')
    else:
        task_to_cluster = mapping_raw
        K = None
    if not isinstance(task_to_cluster, dict):
        raise ValueError(
            f'mapping JSON has unexpected shape: keys={list(mapping_raw)[:5]}')

    # 3. Compute per-task Δ + win/loss
    common = set(ours['per_task']) & set(hydra['per_task'])
    if not common:
        raise RuntimeError(
            f'no overlap between ours ({len(ours["per_task"])}) and '
            f'hydra ({len(hydra["per_task"])}) per_task keys')
    print(f'[fisher] {len(common)} common tasks between ours + hydra')

    rows = []  # (task_id, cluster_id, delta, label)
    for tid in sorted(common):
        rouge_ours = ours['per_task'][tid]['rougeL']
        rouge_hydra = hydra['per_task'][tid]['rougeL']
        delta = rouge_ours - rouge_hydra
        if abs(delta) <= args.tie_eps:
            label = 'tie'
        elif delta > 0:
            label = 'ours'
        else:
            label = 'hydra'
        cid = task_to_cluster.get(tid)
        if cid is None:
            print(f'  WARN: task {tid} has no cluster_id in mapping; skipping')
            continue
        rows.append((tid, int(cid), delta, label))

    n = len(rows)
    print(f'[fisher] {n} tasks with cluster_id assignment')

    # 4. Aggregate counts per cluster
    cluster = defaultdict(lambda: {'ours': 0, 'hydra': 0, 'tie': 0, 'n': 0})
    overall = {'ours': 0, 'hydra': 0, 'tie': 0}
    for _, cid, _, label in rows:
        cluster[cid][label] += 1
        cluster[cid]['n'] += 1
        overall[label] += 1
    K = K or (max(cluster) + 1)

    # 5. Statistical tests
    import numpy as np
    try:
        from scipy.stats import chi2_contingency, fisher_exact
    except ImportError:
        raise ImportError('scipy required: pip install scipy') from None

    cids = sorted(cluster)
    # 5a. 2x|cluster| chi-square: ours-win vs (hydra+tie), grouped by cluster
    table_2way = np.array([
        [cluster[c]['ours'], cluster[c]['hydra'] + cluster[c]['tie']]
        for c in cids
    ])
    chi2_2, p_2, dof_2, _ = chi2_contingency(table_2way)

    # 5b. 3x|cluster| chi-square: ours / hydra / tie × cluster
    table_3way = np.array([
        [cluster[c]['ours'], cluster[c]['hydra'], cluster[c]['tie']]
        for c in cids
    ])
    # Drop zero-only columns to avoid degeneracy
    nonzero_cols = [j for j in range(table_3way.shape[1]) if table_3way[:, j].sum() > 0]
    table_3way_used = table_3way[:, nonzero_cols]
    if table_3way_used.shape[1] >= 2:
        chi2_3, p_3, dof_3, _ = chi2_contingency(table_3way_used)
    else:
        chi2_3 = p_3 = dof_3 = None

    # 5c. Per-cluster Fisher exact: this cluster's ours-win rate vs all-other-clusters'
    fisher_per_cluster = {}
    total_ours = sum(c['ours'] for c in cluster.values())
    total_other = n - total_ours
    for cid in cids:
        in_ours = cluster[cid]['ours']
        in_total = cluster[cid]['n']
        in_other = in_total - in_ours
        out_ours = total_ours - in_ours
        out_other = total_other - in_other
        if min(in_total, n - in_total) >= 1:
            odds, p = fisher_exact([[in_ours, in_other], [out_ours, out_other]])
            fisher_per_cluster[cid] = {'odds_ratio': float(odds), 'p_value': float(p),
                                        'in_ours': in_ours, 'in_total': in_total}

    # 6. Top clusters for ours / hydra by win rate
    cluster_rates = []
    for c in cids:
        if cluster[c]['n'] >= 3:  # min sample for meaningful rate
            rate_ours = cluster[c]['ours'] / cluster[c]['n']
            rate_hydra = cluster[c]['hydra'] / cluster[c]['n']
            cluster_rates.append((c, rate_ours, rate_hydra, cluster[c]['n']))
    top_ours = sorted(cluster_rates, key=lambda x: -x[1])[:5]
    top_hydra = sorted(cluster_rates, key=lambda x: -x[2])[:5]

    # 7. Output
    payload = {
        'n_tasks': n,
        'tie_eps': args.tie_eps,
        'mean_delta': float(np.mean([r[2] for r in rows])),
        'wins_ours': overall['ours'],
        'wins_hydra': overall['hydra'],
        'ties': overall['tie'],
        'cluster_breakdown': {int(c): cluster[c] for c in cids},
        'tests': {
            'chi2_2way_ours_vs_other_by_cluster': {
                'stat': float(chi2_2), 'p_value': float(p_2),
                'dof': int(dof_2), 'shape': list(table_2way.shape),
            },
            'chi2_3way_ours_hydra_tie_by_cluster': (
                {'stat': float(chi2_3), 'p_value': float(p_3),
                 'dof': int(dof_3), 'shape': list(table_3way_used.shape)}
                if chi2_3 is not None else None),
            'fisher_per_cluster': {int(k): v for k, v in fisher_per_cluster.items()},
        },
        'top_5_clusters_ours_wins': [{'cluster_id': c, 'ours_rate': r_o,
                                       'hydra_rate': r_h, 'n_tasks': n}
                                      for c, r_o, r_h, n in top_ours],
        'top_5_clusters_hydra_wins': [{'cluster_id': c, 'ours_rate': r_o,
                                        'hydra_rate': r_h, 'n_tasks': n}
                                       for c, r_o, r_h, n in top_hydra],
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    print()
    print(f'  Overall: ours wins {overall["ours"]}, hydra wins '
          f'{overall["hydra"]}, ties {overall["tie"]} (mean Δ={payload["mean_delta"]:+.4f})')
    print(f'  χ² 2-way (ours vs other × cluster): χ²={chi2_2:.2f}, p={p_2:.4f}')
    if chi2_3 is not None:
        print(f'  χ² 3-way: χ²={chi2_3:.2f}, p={p_3:.4f}')
    print(f'  Fisher per-cluster significant (p<0.05): '
          f'{[c for c, v in fisher_per_cluster.items() if v["p_value"] < 0.05]}')
    print(f'\n[fisher] wrote {args.out}')


if __name__ == '__main__':
    main()
