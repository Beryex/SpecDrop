#!/usr/bin/env python3
"""Paper-introduction specialization figure: 4 settings × 2 methods.

Layout: 2 rows × 4 cols (no per-panel titles beyond the column headers).
  Top row    : ours (per-category × per-branch ablation Δ)
  Bottom row : architecture-matched No-Routing control (uniform 1/K mask; +SE wherever ours has one)
  Cols (L→R) : CIFAR-100 (K=20) | ImageNet ViT (K=46) | SlimPajama (K=7) | LoRA SuperNI (K=20)

Real data sources (preferred):
  CIFAR ours / no-routing : outputs/analysis/specialization/{ours,no_routing}_s42.json
  ViT ours / no-routing   : outputs/analysis/vit_diag/{ours_vit, mbvit_no_routing_se}_s42.json
  NLP ours / no-routing   : outputs/analysis/nlp_diag/{ours_phaseP, no_routing_se05}_s42.json
                            (falls back to phaseP_pa0.6_wr1.0_s42 for ours)
  LoRA ours / no-routing  : outputs/analysis/lora_diag/{ours, mb_lora_no_routing_se1.0}_s42.json

All eight diagnostic JSONs are required (a missing file raises
FileNotFoundError); they are produced by scripts/experiments/alignment/run.sh.
The diag-argmax badges are computed from the same seed-42 JSONs (CIFAR:
most-sensitive branch per superclass, ties to the lowest branch index;
ViT/NLP/LoRA: the `diag_hits` field written by the diagnose scripts).

Sign / magnitude normalization (per-panel):
  Convert every Δ to "ablation importance" = |Δ|, max-normalize per column
  (so ours and its baseline share a vmin=0, vmax=max common to that column).

Usage:
  python scripts/plot_intro_specialization.py \\
         --output outputs/analysis/fig_intro_specialization   # writes .pdf and .png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─── Data loaders (returns 2-D numpy array, |Δ|, rows=cat, cols=branch) ─────
def load_cifar(setting: str) -> np.ndarray:
    """setting ∈ {'ours', 'no_routing'}."""
    p = f'outputs/analysis/specialization/{setting}_s42.json'
    d = json.load(open(p))
    # kd_matrix is M×K as written by evaluation.metrics._compute_pruning_sensitivity
    # (kd_matrix[c, k]): rows = categories, cols = branches — already the paper
    # convention, so no transpose.
    kd = np.array(d['pruning_sensitivity']['kd_matrix'])
    return np.abs(kd)


def _load_vit_diag(path: str) -> np.ndarray:
    d = json.load(open(path))
    return np.abs(np.array(d['delta_matrix']))


def _load_nlp_diag(path: str) -> np.ndarray:
    """Build (n_domains × n_branches) |Δ| = |ablated_ppl − baseline_ppl|."""
    d = json.load(open(path))
    bp = {int(k): v for k, v in d['baseline_ppl_per_domain'].items()}
    ap = {int(k): {int(kk): vv for kk, vv in v.items()}
          for k, v in d['ablated_ppl_per_domain'].items()}
    K = d['config_summary']['num_branches']
    domains = sorted(bp.keys())
    delta = np.zeros((len(domains), K))
    for i, dd in enumerate(domains):
        for k in range(K):
            ablated = ap.get(k, {}).get(dd, bp[dd])
            delta[i, k] = ablated - bp[dd]
    return np.abs(delta)


def _load_lora_diag(path: str) -> np.ndarray:
    d = json.load(open(path))
    return np.abs(np.array(d['delta_matrix']))


def load_vit_ours() -> np.ndarray:
    return _load_vit_diag('outputs/analysis/vit_diag/ours_vit_s42.json')


def load_vit_no_routing() -> np.ndarray:
    """Architecture-matched ViT no-routing (+SE, as ours) diagnostic."""
    return _load_vit_diag('outputs/analysis/vit_diag/mbvit_no_routing_se_s42.json')


def _nlp_ours_path() -> str:
    """Current name first, then the legacy `phaseP_pa0.6_wr1.0` name."""
    for p in ('outputs/analysis/nlp_diag/ours_phaseP_s42.json',
              'outputs/analysis/nlp_diag/phaseP_pa0.6_wr1.0_s42.json'):
        if os.path.exists(p):
            return p
    raise FileNotFoundError('NLP ours diag JSON not found')


def load_nlp_ours() -> np.ndarray:
    return _load_nlp_diag(_nlp_ours_path())


def load_nlp_no_routing() -> np.ndarray:
    return _load_nlp_diag('outputs/analysis/nlp_diag/no_routing_se05_s42.json')


def load_lora_ours() -> np.ndarray:
    return _load_lora_diag('outputs/analysis/lora_diag/ours_s42.json')


def load_lora_no_routing() -> np.ndarray:
    # +SE, as ours
    return _load_lora_diag('outputs/analysis/lora_diag/mb_lora_no_routing_se1.0_s42.json')


# ─── Plot ──────────────────────────────────────────────────────────────────
COL_LABELS = ['CIFAR-100 — aligned\n(K=20 superclasses)',
              'ImageNet ViT — aligned\n(K=46 supercategories)',
              'SlimPajama — fuzzy\n(K=7 domains)',
              'LoRA SuperNI — fuzzy\n(K=20 task clusters)']
ROW_LABELS = ['Ours', 'No-Routing\ncontrol']

def _json_badge(path: str, total_key: str) -> tuple[int, int]:
    d = json.load(open(path))
    return int(d['diag_hits']), int(d[total_key])


def diag_hits() -> list[tuple[int, int]]:
    """Seed-42 diag-argmax badges (hits, categories) for the four ours panels.

    CIFAR: superclasses whose most pruning-sensitive branch is the round-robin
    assigned branch c mod K (np.argmax breaks ties to the lowest branch index,
    as Tab. 1). ViT / NLP / LoRA: the `diag_hits` field of the diag JSONs
    (NLP counts the domains with validation coverage; LoRA the clusters with
    held-out test tasks).
    """
    kd = load_cifar('ours')
    M, K = kd.shape
    cifar = int(sum(int(np.argmax(kd[c])) == c % K for c in range(M)))
    return [(cifar, M),
            _json_badge('outputs/analysis/vit_diag/ours_vit_s42.json', 'n_cats'),
            _json_badge(_nlp_ours_path(), 'n_domains'),
            _json_badge('outputs/analysis/lora_diag/ours_s42.json', 'n_cats')]


def per_row_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize each row independently to [0, 1] by row max.

    NOTE: kept as a utility but NOT used by plot_grid as of v3 — per-row
    normalization saturates the no-routing baseline panels (each row's
    max ≈ 1 even for uniform-noise data, making baseline look as red as
    ours). The figure now uses `paired_normalize` to preserve absolute
    magnitude contrast between ours and baseline within each setting.
    """
    m = np.asarray(mat, dtype=np.float64)
    m = np.maximum(m, 0)
    row_max = m.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    return m / row_max


def paired_normalize(ours: np.ndarray, baseline: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-setting normalization keyed on `ours.max()`.

    Both ours and baseline divide by ours' max within the setting, then
    clip to [0, 1]. Rationale:
    - ours panel ALWAYS reaches saturation=1 → consistent visual scale
      across the 4 settings (CIFAR / ViT / NLP / LoRA all show the same
      "ours hot pixels are full-red" baseline expectation).
    - baseline preserves absolute magnitude relative to ours: where
      baseline cells fall below ours' max they look proportionally
      faint (which is the typical case — no specialization → small Δ);
      a baseline cell above ours' max would clip to saturation=1 (none
      does in the paper figure).
    - Each setting normalizes independently (no cross-setting shared
      max — would crush low-ratio settings against high-ratio ones).

    Returns (ours_normalized, baseline_normalized), both in [0, 1].
    """
    ours_n = np.maximum(np.asarray(ours, dtype=np.float64), 0)
    base_n = np.maximum(np.asarray(baseline, dtype=np.float64), 0)
    ref_max = float(ours_n.max())
    if ref_max <= 0:
        return ours_n, base_n
    return np.clip(ours_n / ref_max, 0, 1), np.clip(base_n / ref_max, 0, 1)


def plot_grid(out_base: str):
    cifar_o, cifar_n = load_cifar('ours'), load_cifar('no_routing')
    vit_o, vit_n = load_vit_ours(), load_vit_no_routing()
    nlp_o, nlp_n = load_nlp_ours(), load_nlp_no_routing()
    lora_o, lora_n = load_lora_ours(), load_lora_no_routing()
    badges = diag_hits()
    print(f'  diag-argmax badges: {badges}')

    cols = [(cifar_o, cifar_n), (vit_o, vit_n), (nlp_o, nlp_n), (lora_o, lora_n)]

    # 2 rows × 4 cols. Ribbon-wide (3.25:1) for intro figure: the common
    # convention for multi-panel hero/teaser figures sits at 2.5:1–3.5:1.
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 3.63),
                              gridspec_kw={'wspace': 0.20, 'hspace': 0.10,
                                            'left': 0.07, 'right': 0.985,
                                            'top': 0.86, 'bottom': 0.04})

    for c, (ours_mat, base_mat) in enumerate(cols):
        # Per-setting PAIRED normalize: ours and baseline share a max within
        # the column. Ensures baseline panels look light (faint) when their
        # absolute Δ is small relative to ours' diagonal, instead of being
        # saturated by per-row normalize. Cross-setting independent — keeps
        # per-setting contrast intact (ViT's strong diagonal and the weak
        # SuperNI one are each readable in their own panels).
        ours_n, base_n = paired_normalize(ours_mat, base_mat)
        for r, mat_norm in enumerate([ours_n, base_n]):
            ax = axes[r, c]
            ax.imshow(mat_norm, cmap='OrRd', vmin=0, vmax=1,
                       aspect='auto', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color('#666')
            if r == 0:
                ax.set_title(COL_LABELS[c], fontsize=15, pad=8)
                # Top-right diag-argmax annotation on each ours panel (the
                # diagonal ends bottom-right, so the badge must not sit there).
                hits, total = badges[c]
                ax.text(0.975, 0.97, f'diag: {hits}/{total}',
                          transform=ax.transAxes, ha='right', va='top',
                          fontsize=16, color='black',
                          bbox=dict(boxstyle='round,pad=0.25',
                                     facecolor='white', edgecolor='#888',
                                     alpha=0.85, linewidth=0.6))
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=15, fontweight='bold',
                                rotation=0, labelpad=52, va='center')

    # Save
    out_pdf = f'{out_base}.pdf'
    out_png = f'{out_base}.png'
    os.makedirs(os.path.dirname(out_base) or '.', exist_ok=True)
    # dpi=600 embeds each heatmap at ~6 px per cell so cell edges are even
    # (the default 100 dpi gave 3-5 px cells on the 46x46 ViT panel).
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05, dpi=600)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f'[plot_intro_specialization] wrote {out_pdf}')
    print(f'[plot_intro_specialization] wrote {out_png}')

    # Sanity diagnostics (helpful for the handoff)
    def _diag_off_ratio(m):
        if m.shape[0] != m.shape[1]:
            return float('nan')
        return float(m.diagonal().mean() / max(1e-12,
                      (m.sum() - m.trace()) / (m.size - m.shape[0])))
    print()
    print('  diag/off ratio:')
    for c, lbl in enumerate(['CIFAR', 'ViT', 'NLP', 'LoRA']):
        ours_mat, base_mat = cols[c]
        print(f'    {lbl:6s} ours={_diag_off_ratio(ours_mat):.2f}  '
              f'baseline={_diag_off_ratio(base_mat):.2f}  '
              f'shape={ours_mat.shape}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', default='outputs/analysis/fig_intro_specialization',
                    help='output path without extension (.pdf and .png are written)')
    args = ap.parse_args()
    out = args.output
    for ext in ('.pdf', '.png'):
        if out.endswith(ext):
            out = out[:-len(ext)]
    plot_grid(out)


if __name__ == '__main__':
    main()
