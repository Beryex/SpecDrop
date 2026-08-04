"""Aggregate per-cell alignment data into a 4-setting × method table.

Reads pruning sensitivity JSON files produced by:
  scripts/diagnose_cifar_specialization.py  → outputs/analysis/specialization/
  scripts/diagnose_vit_specialization.py    → outputs/analysis/vit_diag/
  scripts/diagnose_nlp_specialization.py    → outputs/analysis/nlp_diag/
  scripts/diagnose_lora_specialization.py   → outputs/analysis/lora_diag/

For each (setting, method, seed), extracts:
  • diag-argmax hits / K  ("alignment fraction")
  • diag/off magnitude ratio
  • max|Δ|

Then aggregates across 3 seeds → mean ± std and writes paper-ready tables:
  outputs/analysis/alignment_table.{md,json,tex}

Cross-script schema differences are handled by per-setting parsers — each
sets a (M=cats, K=branches) numpy delta matrix and a method/seed key.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import numpy as np


# ─── Per-setting JSON parsers ──────────────────────────────────────────────

def _alignment_fraction(kd_matrix, n_diag=None):
    """Diag-argmax hits / n_diag for a (rows=cats, cols=branches) or
    (rows=branches, cols=cats) matrix; we always take min(R, C) as n_diag.
    Hits = number of c where argmax over OTHER axis == c.
    Caller must pass kd_matrix in convention (rows=cats, cols=branches)."""
    kd = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    R, C = kd.shape
    n = n_diag if n_diag is not None else min(R, C)
    if n == 0:
        return 0, 0
    hits = sum(int(kd[c].argmax() == c) for c in range(n))
    return hits, n


def _diag_off_ratio(kd_matrix):
    kd = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    R, C = kd.shape
    n = min(R, C)
    if n == 0:
        return float('nan')
    diag_v = np.array([kd[c, c] for c in range(n)])
    off_v = np.array([kd[r, c] for r in range(R) for c in range(C)
                       if r < n and c < n and r != c])
    if off_v.mean() <= 0:
        return float('inf') if diag_v.mean() > 0 else float('nan')
    return float(diag_v.mean() / off_v.mean())


def _seed_from_path(p):
    m = re.search(r'_s(\d+)(?:\.json|/|$)', p)
    if m:
        return int(m.group(1))
    m = re.search(r'/s(\d+)/', p)  # e.g., outputs/cv_*/s42/results.json
    if m:
        return int(m.group(1))
    return None


def _method_from_basename(p):
    base = os.path.basename(p).replace('.json', '')
    return re.sub(r'_s\d+$', '', base)


def parse_cifar_json(path):
    """CIFAR JSON has either:
    (a) new diagnose_cifar (rtx5090_15) — stores
        `alignment_diag_argmax_hits` + `alignment_n_diag` directly.
    (b) legacy analyze_e1/e2 cache — only `pruning_sensitivity::kd_matrix`,
        recompute via argmax|kd_mk[c]| over branches (no missing cats).
    kd_matrix is (K, M); transpose to (M, K) — paper convention."""
    d = json.load(open(path))
    ps = d.get('pruning_sensitivity', d)
    kd_km = np.array(ps['kd_matrix'])  # (K, M)
    kd_mk = kd_km.T                     # (M, K)
    if 'alignment_diag_argmax_hits' in d and 'alignment_n_diag' in d:
        hits = int(d['alignment_diag_argmax_hits'])
        n_diag = int(d['alignment_n_diag'])
    else:
        hits, n_diag = _alignment_fraction(kd_mk)
    return {
        'kd_matrix': kd_mk,
        'alignment_hits': hits,
        'alignment_n': n_diag,
        'diag_off_ratio': _diag_off_ratio(kd_mk),
        'max_abs_delta': float(np.abs(kd_mk).max()),
    }


def parse_vit_json(path):
    """ViT diagnose output: stores `diag_hits` + `n_cats` (= M=46 BREEDS
    supercats). Use stored values — paper-canonical (argmin signed Δ
    convention; consistent for top-1 since ablation strictly hurts)."""
    d = json.load(open(path))
    kd = np.array(d['delta_matrix'])
    if 'diag_hits' in d and ('n_cats' in d or 'n_domains' in d):
        hits = int(d['diag_hits'])
        n_diag = int(d.get('n_cats', d.get('n_domains')))
    else:
        hits, n_diag = _alignment_fraction(kd)
    return {
        'kd_matrix': kd,
        'alignment_hits': hits,
        'alignment_n': n_diag,
        'diag_off_ratio': _diag_off_ratio(kd),
        'max_abs_delta': float(np.abs(kd).max()),
    }


def parse_nlp_json(path):
    """NLP diagnose output: stores `diag_hits` + `n_domains` (= 6, Book
    domain omitted). Use stored — recompute path mishandles gapped cats."""
    d = json.load(open(path))
    if 'delta_matrix' in d:
        kd = np.array(d['delta_matrix'])
    else:
        bp = {int(k): v for k, v in d['baseline_ppl_per_domain'].items()}
        ap = {int(k): {int(kk): vv for kk, vv in v.items()}
              for k, v in d['ablated_ppl_per_domain'].items()}
        cats = sorted(bp.keys())
        K = (max(max(ap[c].keys()) for c in ap if ap[c]) + 1) if ap else 0
        kd = np.zeros((len(cats), K))
        for ri, c in enumerate(cats):
            for k in range(K):
                kd[ri, k] = ap.get(k, {}).get(c, bp[c]) - bp[c]
    if 'diag_hits' in d and ('n_domains' in d or 'n_cats' in d):
        hits = int(d['diag_hits'])
        n_diag = int(d.get('n_domains', d.get('n_cats')))
    else:
        # Fallback: recompute respecting actual cat ids when available.
        bp = d.get('baseline_ppl_per_domain', {})
        if bp:
            cats = sorted(int(k) for k in bp.keys())
            ap = {int(k): {int(kk): vv for kk, vv in v.items()}
                  for k, v in d.get('ablated_ppl_per_domain', {}).items()}
            hits = 0
            for c in cats:
                deltas = [ap.get(k, {}).get(c, bp[str(c)] if str(c) in bp else bp[c])
                          - (bp[str(c)] if str(c) in bp else bp[c])
                          for k in sorted(ap.keys())]
                if not deltas:
                    continue
                worst_k = sorted(ap.keys())[int(np.argmax(deltas))]
                if worst_k == c:
                    hits += 1
            n_diag = len(cats)
        else:
            hits, n_diag = _alignment_fraction(kd)
    return {
        'kd_matrix': kd,
        'alignment_hits': hits,
        'alignment_n': n_diag,
        'diag_off_ratio': _diag_off_ratio(kd),
        'max_abs_delta': float(np.abs(kd).max()),
    }


def parse_lora_json(path):
    """LoRA merged JSON: 'delta_matrix' of (n_cats, K_branches) +
    'cats' list of cluster_ids. Stores `diag_hits` (argmin signed Δ
    convention since ROUGE can move either direction). Use stored —
    paper §5.6/E.13 cite stored `diag_hits` directly."""
    d = json.load(open(path))
    if d.get('partial', False):
        return None  # shard-only, must be merged first
    kd = np.array(d['delta_matrix'])
    cats = d.get('cats', list(range(kd.shape[0])))
    if 'diag_hits' in d and 'n_cats' in d:
        hits = int(d['diag_hits'])
        n_diag = int(d['n_cats'])
    else:
        # Fallback (legacy/synthetic only): argmax|Δ|. Real diag JSONs
        # store diag_hits computed via argmin(signed Δ) per the LoRA
        # diag script — paper §5.6/E.13's canonical definition.
        hits = 0
        for ri, c in enumerate(cats):
            if 0 <= c < kd.shape[1] and int(np.abs(kd[ri]).argmax()) == c:
                hits += 1
        n_diag = len(cats)
    return {
        'kd_matrix': kd,
        'alignment_hits': hits,
        'alignment_n': n_diag,
        'diag_off_ratio': _diag_off_ratio_lora(kd, cats),
        'max_abs_delta': float(np.abs(kd).max()),
    }


def _diag_off_ratio_lora(kd_matrix, cats):
    kd = np.abs(np.asarray(kd_matrix, dtype=np.float64))
    diag_v = np.array([kd[ri, c] for ri, c in enumerate(cats)
                        if 0 <= c < kd.shape[1]])
    off_v = np.array([kd[ri, k] for ri, c in enumerate(cats)
                       for k in range(kd.shape[1]) if k != c])
    if off_v.size == 0 or off_v.mean() <= 0:
        return float('inf') if diag_v.size and diag_v.mean() > 0 else float('nan')
    return float(diag_v.mean() / off_v.mean())


# ─── Settings table ────────────────────────────────────────────────────────

SETTINGS = [
    ('cifar', 'outputs/analysis/specialization', '*.json', parse_cifar_json),
    ('vit',   'outputs/analysis/vit_diag',       '*.json', parse_vit_json),
    ('nlp',   'outputs/analysis/nlp_diag',       '*.json', parse_nlp_json),
    ('lora',  'outputs/analysis/lora_diag',      '*.json', parse_lora_json),
]


def collect():
    """Walk each setting's analysis dir and group by (setting, method, seed)."""
    by_method = defaultdict(list)  # (setting, method) -> [{seed, alignment_*, ...}]
    for setting, base_dir, pat, parser in SETTINGS:
        if not os.path.isdir(base_dir):
            continue
        for path in sorted(glob.glob(os.path.join(base_dir, pat))):
            if '.shard' in os.path.basename(path):
                continue  # skip LoRA shard files; only read merged
            seed = _seed_from_path(path)
            if seed is None:
                continue
            method = _method_from_basename(path)
            try:
                rec = parser(path)
            except Exception as e:
                print(f'  WARN: {path}: {e}')
                continue
            if rec is None:
                continue
            rec['seed'] = seed
            rec['path'] = path
            by_method[(setting, method)].append(rec)
    return by_method


def aggregate(by_method):
    """For each (setting, method): mean ± std of alignment fraction +
    diag/off ratio, plus alignment_hits formatted as 'X/N (mean)' with σ."""
    summary = {}
    for (setting, method), recs in by_method.items():
        if not recs:
            continue
        n = len(recs)
        hits = [r['alignment_hits'] for r in recs]
        N = recs[0]['alignment_n']
        # All seeds should have same N (e.g. K=20 CIFAR, K=46 ViT, etc.).
        # For LoRA, N might differ across seeds (different held-out cluster counts).
        Ns = sorted({r['alignment_n'] for r in recs})
        ratios = [r['diag_off_ratio'] for r in recs if np.isfinite(r['diag_off_ratio'])]
        max_d = [r['max_abs_delta'] for r in recs]
        seeds = sorted({r['seed'] for r in recs})

        summary[(setting, method)] = {
            'n_seeds': n,
            'seeds': seeds,
            'alignment_n': N,
            'alignment_n_per_seed': Ns,
            'alignment_hits_mean': statistics.mean(hits),
            'alignment_hits_std': statistics.stdev(hits) if n > 1 else 0.0,
            'alignment_frac_mean': statistics.mean(h / r['alignment_n']
                                                     for h, r in zip(hits, recs)),
            'alignment_frac_std': (statistics.stdev(h / r['alignment_n']
                                                       for h, r in zip(hits, recs))
                                    if n > 1 else 0.0),
            'diag_off_ratio_mean': statistics.mean(ratios) if ratios else float('nan'),
            'diag_off_ratio_std': statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            'max_abs_delta_mean': statistics.mean(max_d),
            'max_abs_delta_std': statistics.stdev(max_d) if n > 1 else 0.0,
        }
    return summary


def render_md(summary):
    lines = ['# Branch-category alignment table',
             '',
             'Per-setting × per-method specialization metrics from pruning '
             'sensitivity (zero-ablate branch k, measure metric drop on category c).',
             '`alignment` = (diag-argmax hits) / (n_diag = min(M, K) covered cats).',
             '`diag/off` = mean(|delta|) on diagonal / off-diagonal cells.',
             '']
    by_setting = defaultdict(list)
    for (setting, method), s in summary.items():
        by_setting[setting].append((method, s))
    titles = {
        'cifar': 'CIFAR-100 (ResNet-110 K=20, 200ep)',
        'vit':   'ImageNet ViT-S/16 (K=46 BREEDS, 100ep)',
        'nlp':   'SlimPajama 30M Transformer (K=7, 500M tokens, 10ep)',
        'lora':  'SuperNI LoRA on Llama-3.2-1B (K=20, 3ep)',
    }
    for setting in ['cifar', 'vit', 'nlp', 'lora']:
        if setting not in by_setting:
            continue
        lines.append(f'## {titles[setting]}')
        lines.append('')
        lines.append('| Method | n | Alignment (X/K) ↑ | Frac ↑ | diag/off ↑ | max\\|Δ\\| |')
        lines.append('|---|---|---|---|---|---|')
        for method, s in sorted(by_setting[setting], key=lambda kv: -kv[1]['alignment_frac_mean']):
            N = s['alignment_n']
            hits_str = f'{s["alignment_hits_mean"]:.1f} ± {s["alignment_hits_std"]:.1f} / {N}'
            frac_str = f'{s["alignment_frac_mean"]:.3f} ± {s["alignment_frac_std"]:.3f}'
            ratio_str = (f'{s["diag_off_ratio_mean"]:.2f} ± {s["diag_off_ratio_std"]:.2f}'
                          if s['n_seeds'] > 1 else f'{s["diag_off_ratio_mean"]:.2f}')
            md_str = f'{s["max_abs_delta_mean"]:.3f} ± {s["max_abs_delta_std"]:.3f}'
            lines.append(f'| {method} | {s["n_seeds"]} | {hits_str} | {frac_str} | {ratio_str} | {md_str} |')
        lines.append('')
    return '\n'.join(lines)


def render_tex(summary):
    """LaTeX-ready snippet (booktabs)."""
    lines = ['% Auto-generated by scripts/aggregate_alignment.py']
    by_setting = defaultdict(list)
    for (setting, method), s in summary.items():
        by_setting[setting].append((method, s))
    for setting in ['cifar', 'vit', 'nlp', 'lora']:
        if setting not in by_setting:
            continue
        lines += ['', f'% --- {setting} ---',
                   r'\begin{tabular}{lcccc}',
                   r'  \toprule',
                   r'  Method & Alignment & Frac & diag/off & $\max|\Delta|$ \\',
                   r'  \midrule']
        for method, s in sorted(by_setting[setting], key=lambda kv: -kv[1]['alignment_frac_mean']):
            N = s['alignment_n']
            hit = f'{s["alignment_hits_mean"]:.1f}/{N}'
            frac = f'{s["alignment_frac_mean"]:.3f}'
            ratio = f'{s["diag_off_ratio_mean"]:.2f}$\\times$'
            md = f'{s["max_abs_delta_mean"]:.3f}'
            safe = method.replace('_', '\\_')
            lines.append(f'  {safe} & {hit} & {frac} & {ratio} & {md} \\\\')
        lines += [r'  \bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out_md',   default='outputs/analysis/alignment_table.md')
    ap.add_argument('--out_json', default='outputs/analysis/alignment_table.json')
    ap.add_argument('--out_tex',  default='outputs/analysis/alignment_table.tex')
    args = ap.parse_args()

    print('[aggregate_alignment] scanning per-setting JSONs...')
    by_method = collect()
    summary = aggregate(by_method)

    # Print a compact console table.
    print()
    print(f'{"Setting":<10} {"Method":<26} {"n":<3} {"Hits":<14} {"Frac":<14} {"diag/off"}')
    print('-' * 90)
    for (setting, method), s in sorted(summary.items()):
        hits_str = f'{s["alignment_hits_mean"]:.1f}±{s["alignment_hits_std"]:.1f} /{s["alignment_n"]}'
        frac_str = f'{s["alignment_frac_mean"]:.3f}±{s["alignment_frac_std"]:.3f}'
        ratio_str = f'{s["diag_off_ratio_mean"]:.2f}±{s["diag_off_ratio_std"]:.2f}x'
        print(f'{setting:<10} {method:<26} {s["n_seeds"]:<3} {hits_str:<14} {frac_str:<14} {ratio_str}')

    # JSON
    json_out = {f'{s}::{m}': v for (s, m), v in summary.items()}
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f'\n[aggregate_alignment] wrote {args.out_json}')

    # MD
    with open(args.out_md, 'w') as f:
        f.write(render_md(summary))
    print(f'[aggregate_alignment] wrote {args.out_md}')

    # LaTeX
    with open(args.out_tex, 'w') as f:
        f.write(render_tex(summary))
    print(f'[aggregate_alignment] wrote {args.out_tex}')


if __name__ == '__main__':
    main()
