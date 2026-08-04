#!/usr/bin/env python3
"""T3.1 — Wall-clock + per-method efficiency table from existing results.json.

Walks all 4-setting results.json files and assembles a method × wall-time
table for the paper appendix. LoRA wall-time is not stored in results.json
(trainer_lora.py gap), so we recover it by parsing tqdm completion stamps
in the per-cell training logs.

Outputs (markdown + JSON + LaTeX):
  outputs/analysis/wall_clock_table.md
  outputs/analysis/wall_clock_table.json
  outputs/analysis/wall_clock_table.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


# ─── Setting registry ──────────────────────────────────────────────────────
# (display_name, results_dir, baseline_method_name, log_fallback)
SETTINGS = [
    ('cifar100',       'outputs/rtx5090_cifar100_faithful',     'resnet110',          False),
    ('vit_imagenet',   'outputs/rtx5090_imagenet_vit_faithful', 'vit_small',          False),
    ('nlp_slimpajama', 'outputs/rtx5090_nlp_faithful',          'dense',              False),
    ('lora_superni',   'outputs/rtx5090_lora_faithful',         'single_lora_r320',   True),
]

SEED_SUFFIX = re.compile(r'_s(?:42|123|456)$')


# ─── Helpers ───────────────────────────────────────────────────────────────
def _method_name(cell_dir):
    return SEED_SUFFIX.sub('', os.path.basename(os.path.normpath(cell_dir)))


def _seed_from_cell(cell_dir):
    m = re.search(r'_s(\d+)$', os.path.basename(os.path.normpath(cell_dir)))
    return int(m.group(1)) if m else None


def _walltime_from_results(rj):
    """Best-effort extract wall time in seconds from results.json."""
    cmp = rj.get('compute', {}) if isinstance(rj.get('compute'), dict) else {}
    for k in ('total_training_time_sec', 'total_train_time_sec',
              'wall_time_sec', 'total_time_sec'):
        if k in cmp and cmp[k] is not None:
            return float(cmp[k])
    for k in ('total_training_time_sec', 'wall_time_sec'):
        if k in rj and rj[k] is not None:
            return float(rj[k])
    if rj.get('total_time_h') is not None:
        return float(rj['total_time_h']) * 3600
    return None


def _walltime_from_log(log_path):
    """Recover total training wall-time by parsing the LAST tqdm 100% line per
    epoch. tqdm format: `epoch X/N: 100%|...|n/n [HH:MM:SS<00:00, ...s/it]`.

    Returns total seconds across all epochs, or None if no completion lines.
    """
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
    except Exception:
        return None
    pat = re.compile(
        r'epoch (\d+)/(\d+): 100%\|[^|]*\|\s*(\d+)/\3 '
        r'\[(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})<'
    )
    matches = pat.findall(content)
    if not matches:
        return None
    # Keep last reading per epoch (tqdm overwrites; last wins).
    by_ep = {}
    for ep_idx, _, _, elapsed in matches:
        by_ep[int(ep_idx)] = elapsed
    total_sec = 0.0
    for elapsed in by_ep.values():
        parts = elapsed.split(':')
        if len(parts) == 3:
            h, m, s = (int(p) for p in parts)
            total_sec += h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = (int(p) for p in parts)
            total_sec += m * 60 + s
    return total_sec


def _avg_epoch_sec(rj):
    cmp = rj.get('compute', {}) if isinstance(rj.get('compute'), dict) else {}
    return cmp.get('avg_epoch_time_sec')


def _num_params_from_results(rj):
    cmp = rj.get('compute', {}) if isinstance(rj.get('compute'), dict) else {}
    for k in ('total_params', 'num_params', 'params'):
        if cmp.get(k) is not None:
            return int(cmp[k])
    # LoRA stores trainable params at top level
    if rj.get('num_trainable_params') is not None:
        return int(rj['num_trainable_params'])
    return None


def _num_params_via_build(rj):
    """Last-resort: build the model and count parameters."""
    cfg = rj.get('config')
    if not cfg:
        return None
    try:
        sys.path.insert(0, '.')
        from models import build_model
        m = build_model(cfg)
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return None


def _collect_for_setting(setting_name, base_dir, baseline_name, use_log_fallback):
    """Return {method: [{seed, wall_sec, avg_epoch_sec, n_epochs, params}, ...]}."""
    if not os.path.isdir(base_dir):
        return {}
    per_method = defaultdict(list)
    for rj_path in sorted(glob.glob(f'{base_dir}/*/results.json')):
        cell = os.path.dirname(rj_path)
        method = _method_name(cell)
        seed = _seed_from_cell(cell)
        try:
            with open(rj_path) as f:
                rj = json.load(f)
        except Exception as e:
            print(f'  WARN: failed to load {rj_path}: {e}')
            continue
        wall = _walltime_from_results(rj)
        if wall is None and use_log_fallback:
            log_path = os.path.join(cell, f'{method}_s{seed}.log')
            wall = _walltime_from_log(log_path)
        if wall is None:
            continue
        params = _num_params_from_results(rj) or _num_params_via_build(rj)
        per_method[method].append({
            'seed': seed,
            'wall_sec': wall,
            'avg_epoch_sec': _avg_epoch_sec(rj),
            'n_epochs': (rj.get('compute', {}) or {}).get('num_epochs'),
            'params': params,
        })
    return dict(per_method)


def _summarize(per_method, baseline_method=None):
    """Aggregate to mean ± std across seeds. If baseline_method given, also
    compute paired per-seed ratio (method_wall_si / baseline_wall_si) and
    its mean ± std (more rigorous than mean(method)/mean(baseline)).
    """
    # Build {seed: wall_sec} for baseline lookup.
    baseline_by_seed = {}
    if baseline_method and baseline_method in per_method:
        for r in per_method[baseline_method]:
            if r['seed'] is not None and r['wall_sec'] is not None:
                baseline_by_seed[r['seed']] = r['wall_sec']

    out = {}
    for method, runs in per_method.items():
        walls = [r['wall_sec'] for r in runs if r['wall_sec'] is not None]
        params = [r['params'] for r in runs if r['params'] is not None]
        epochs_vals = [r['avg_epoch_sec'] for r in runs if r['avg_epoch_sec'] is not None]
        n = len(walls)
        if not walls:
            continue
        mean_h = statistics.mean(walls) / 3600
        std_h = statistics.stdev(walls) / 3600 if n > 1 else 0.0

        # Per-seed paired ratios (only seeds present in both method + baseline).
        ratios = []
        for r in runs:
            if r['seed'] in baseline_by_seed and r['wall_sec'] is not None:
                b = baseline_by_seed[r['seed']]
                if b > 0:
                    ratios.append(r['wall_sec'] / b)
        ratio_mean = statistics.mean(ratios) if ratios else None
        ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else (0.0 if ratios else None)

        out[method] = {
            'n_seeds': n,
            'wall_hours_mean': mean_h,
            'wall_hours_std': std_h,
            'ratio_mean': ratio_mean,
            'ratio_std': ratio_std,
            'avg_epoch_sec_mean': statistics.mean(epochs_vals) if epochs_vals else None,
            'n_epochs': runs[0].get('n_epochs'),
            'params_mean': int(statistics.mean(params)) if params else None,
        }
    return out


# ─── Output ────────────────────────────────────────────────────────────────
def _format_table_row_md(method, s, baseline_h):
    wall = (f'{s["wall_hours_mean"]:.2f} ± {s["wall_hours_std"]:.2f}'
            if s['n_seeds'] > 1 else f'{s["wall_hours_mean"]:.2f}')
    if s['ratio_mean'] is None:
        ratio = '—'
    elif s['ratio_std'] is not None and s['n_seeds'] > 1:
        ratio = f'{s["ratio_mean"]:.2f} ± {s["ratio_std"]:.2f}×'
    else:
        ratio = f'{s["ratio_mean"]:.2f}×'
    ep = f'{s["avg_epoch_sec_mean"]:.1f}' if s['avg_epoch_sec_mean'] else '—'
    ne = s['n_epochs'] if s['n_epochs'] else '—'
    p = f'{s["params_mean"]/1e6:.3f} M' if s['params_mean'] else '—'
    return f'| {method} | {s["n_seeds"]} | {wall} | {ratio} | {ep} | {ne} | {p} |'


def _format_table_row_tex(method, s, baseline_h):
    wall = (f'{s["wall_hours_mean"]:.2f} $\\pm$ {s["wall_hours_std"]:.2f}'
            if s['n_seeds'] > 1 else f'{s["wall_hours_mean"]:.2f}')
    if s['ratio_mean'] is None:
        ratio = '--'
    elif s['ratio_std'] is not None and s['n_seeds'] > 1:
        ratio = f'{s["ratio_mean"]:.2f} $\\pm$ {s["ratio_std"]:.2f}$\\times$'
    else:
        ratio = f'{s["ratio_mean"]:.2f}$\\times$'
    p = f'{s["params_mean"]/1e6:.3f}' if s['params_mean'] else '--'
    safe = method.replace('_', '\\_')
    return f'  {safe} & {wall} & {ratio} & {p} \\\\'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out_md',   default='outputs/analysis/wall_clock_table.md')
    ap.add_argument('--out_json', default='outputs/analysis/wall_clock_table.json')
    ap.add_argument('--out_tex',  default='outputs/analysis/wall_clock_table.tex')
    args = ap.parse_args()

    summary = {}
    for setting, base_dir, baseline_name, use_log in SETTINGS:
        per_method = _collect_for_setting(setting, base_dir, baseline_name, use_log)
        agg = _summarize(per_method, baseline_method=baseline_name)
        baseline_h = agg.get(baseline_name, {}).get('wall_hours_mean')
        summary[setting] = {
            'baseline_method': baseline_name,
            'baseline_hours': baseline_h,
            'methods': agg,
        }

    # JSON
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[wall_clock] wrote {args.out_json}')

    # Markdown
    md = ['# Wall-clock Table — 4-setting comparison',
          '',
          'All wall-time numbers are 3-seed mean ± std hours on RTX 5090, '
          'bf16 mixed-precision. `× baseline` is relative to the per-setting '
          'reference method (CIFAR: ResNet-110 dense; ViT: ViT-S/16 dense; '
          'NLP: dense Transformer; LoRA: vanilla single LoRA r=320). '
          'LoRA times recovered from per-cell tqdm logs '
          '(`trainer_lora.py` does not store wall-time in `results.json`).',
          '']

    setting_titles = {
        'cifar100':       'CIFAR-100 (ResNet-110, 200 epochs)',
        'vit_imagenet':   'ImageNet ViT-S/16 (K=46 BREEDS, 100 epochs)',
        'nlp_slimpajama': 'SlimPajama 30M Transformer (500M tokens, 10 epochs)',
        'lora_superni':   'SuperNI LoRA on Llama-3.2-1B (3 epochs)',
    }
    for setting in [s[0] for s in SETTINGS]:
        if setting not in summary:
            continue
        baseline_h = summary[setting]['baseline_hours']
        baseline_name = summary[setting]['baseline_method']
        md.append(f'## {setting_titles.get(setting, setting)}')
        md.append('')
        baseline_disp = (f'`{baseline_name}` = {baseline_h:.2f} h'
                         if baseline_h else f'`{baseline_name}` (missing)')
        md.append(f'Baseline: {baseline_disp}.')
        md.append('')
        md.append('| Method | n | Wall (h) | × baseline | Avg ep (s) | epochs | Params |')
        md.append('|---|---|---|---|---|---|---|')
        rows = sorted(summary[setting]['methods'].items(),
                      key=lambda kv: kv[1]['wall_hours_mean'])
        for method, s in rows:
            md.append(_format_table_row_md(method, s, baseline_h))
        md.append('')
    with open(args.out_md, 'w') as f:
        f.write('\n'.join(md))
    print(f'[wall_clock] wrote {args.out_md}')

    # LaTeX
    tex_lines = ['% Auto-generated by scripts/wall_clock_table.py']
    for setting in [s[0] for s in SETTINGS]:
        if setting not in summary:
            continue
        baseline_h = summary[setting]['baseline_hours']
        baseline_name = summary[setting]['baseline_method']
        tex_lines += [
            '',
            f'% --- {setting_titles.get(setting, setting)} ---',
            r'\begin{tabular}{lccc}',
            r'  \toprule',
            r'  Method & Wall (h) & $\times$ baseline & Trainable (M) \\',
            r'  \midrule',
        ]
        rows = sorted(summary[setting]['methods'].items(),
                      key=lambda kv: kv[1]['wall_hours_mean'])
        for method, s in rows:
            tex_lines.append(_format_table_row_tex(method, s, baseline_h))
        tex_lines += [r'  \bottomrule', r'\end{tabular}']
    with open(args.out_tex, 'w') as f:
        f.write('\n'.join(tex_lines))
    print(f'[wall_clock] wrote {args.out_tex}')


if __name__ == '__main__':
    main()
