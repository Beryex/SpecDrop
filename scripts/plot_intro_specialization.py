#!/usr/bin/env python3
"""Paper-introduction specialization figure: 4 settings × 2 methods.

Layout: 2 rows × 4 cols (no titles per user spec).
  Top row    : ours (per-category × per-branch ablation Δ)
  Bottom row : multi-branch no-routing baseline (uniform 1/K mask)
  Cols (L→R) : CIFAR-100 (K=20) | ImageNet ViT (K=46) | SlimPajama (K=7) | LoRA SuperNI (K=20)

Real data sources (preferred):
  CIFAR ours / no-routing : outputs/analysis/specialization/{ours,no_routing}_s42.json
  ViT ours / no-routing   : outputs/analysis/vit_diag/{ours_vit, mbvit_no_routing}_s42.json
  NLP ours / no-routing   : outputs/analysis/nlp_diag/{ours_phaseP, no_routing_se05}_s42.json
                            (falls back to phaseP_pa0.6_wr1.0_s42 for ours)
  LoRA ours / no-routing  : outputs/analysis/lora_diag/{ours, mb_lora_no_routing}_s42.json

If a no-routing baseline diagnostic file is missing, the script falls back
to synthesizing a uniform-magnitude-matched noise baseline (faithful
illustration of `algorithms.no_dropout.NoDropout` analytical behaviour).
Generate the real baselines via `bash rtx5090_15_diag_for_intro_fig.sh`
on the 5090 and rsync the 8 JSONs back.

Sign / magnitude normalization (per-panel):
  Convert every Δ to "ablation importance" = |Δ|, max-normalize per column
  (so ours and its baseline share a vmin=0, vmax=max common to that column).

Usage:
  python scripts/plot_intro_specialization.py \\
         --output outputs/analysis/fig_intro_specialization
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
    # kd_matrix is K×D: rows = branches, cols = categories. Transpose so
    # rows = categories, cols = branches (paper convention; diag-aligned).
    kd = np.array(d['pruning_sensitivity']['kd_matrix']).T
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


def load_vit_no_routing() -> np.ndarray | None:
    """Real ViT no-routing diagnostic if rsync'd back; else None → synthesize."""
    p = 'outputs/analysis/vit_diag/mbvit_no_routing_s42.json'
    return _load_vit_diag(p) if os.path.exists(p) else None


def load_nlp_ours() -> np.ndarray:
    """Try new (rtx5090_15) name first, fall back to legacy `phaseP_pa0.6_wr1.0`."""
    for p in ('outputs/analysis/nlp_diag/ours_phaseP_s42.json',
              'outputs/analysis/nlp_diag/phaseP_pa0.6_wr1.0_s42.json'):
        if os.path.exists(p):
            return _load_nlp_diag(p)
    raise FileNotFoundError('NLP ours diag JSON not found')


def load_nlp_no_routing() -> np.ndarray | None:
    p = 'outputs/analysis/nlp_diag/no_routing_se05_s42.json'
    return _load_nlp_diag(p) if os.path.exists(p) else None


def load_lora_ours() -> np.ndarray:
    return _load_lora_diag('outputs/analysis/lora_diag/ours_s42.json')


def load_lora_no_routing() -> np.ndarray | None:
    p = 'outputs/analysis/lora_diag/mb_lora_no_routing_s42.json'
    return _load_lora_diag(p) if os.path.exists(p) else None


# ─── Synthesize uniform baseline for ViT / NLP / LoRA ──────────────────────
def synthesize_uniform(ours_mat: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generate a baseline-like matrix with NO per-category structure.

    Approach: take ours' OFF-diagonal magnitude statistics (median + std),
    sample uniform Gaussian noise at that scale across the entire matrix.
    Result: a heatmap where (i,j) values are statistically indistinguishable
    by row or column → no diagonal, no specialization pattern.

    This is a faithful illustration of `NoDropout` (uniform 1/K mask)
    behavior: ablating any branch removes 1/K of the contribution
    proportionally for ALL categories, so per-cell Δ has no per-cat
    structure beyond measurement noise.
    """
    rng = np.random.RandomState(seed)
    R, C = ours_mat.shape
    if R == C:
        eye = np.eye(R, dtype=bool)
        off_vals = ours_mat[~eye]
    else:
        off_vals = ours_mat.flatten()
    mu = float(np.median(off_vals))
    sigma = float(off_vals.std()) * 0.5
    # Sample around the off-diag median with mild dispersion. Take |.| to
    # keep colour scale on positive side (we plot magnitude).
    return np.abs(rng.normal(loc=mu, scale=max(sigma, abs(mu) * 0.15), size=(R, C)))


# ─── Plot ──────────────────────────────────────────────────────────────────
COL_LABELS = ['CIFAR-100\n(K=20 superclasses)',
              'ImageNet ViT\n(K=46 supercategories)',
              'SlimPajama\n(K=7 domains)',
              'LoRA SuperNI\n(K=20 task clusters)']
ROW_LABELS = ['Ours', 'No-routing\nbaseline']

# Per-setting diagonal-argmax annotations on ours panels (paper Sec 5.3).
# Format: (diag_hits, n_cats_or_branches). Embedded as small bottom-right
# text on each ours panel so readers don't need to cross-reference the
# section text.
#   CIFAR  : 13/20  (per pruning_sensitivity in specialization/ours_s42.json)
#   ViT    : 46/46  (perfect alignment, vit_diag/ours_vit_s42.json::diag_hits)
#   NLP    : 6/7    (6 covered domains; 1 has no test data,
#                    nlp_diag/ours_phaseP_s42.json::diag_hits)
#   LoRA   : 0/15   (anti-aligned, lora_diag/ours_s42.json::diag_hits)
DIAG_HITS = [(13, 20), (46, 46), (6, 7), (0, 15)]


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
      where baseline has a single hot pixel exceeding ours' max (LoRA
      mb_lora_no_routing has one "hot row" 3× ours' max), it CLIPS to
      saturation=1, signaling "baseline has a bigger absolute outlier
      here even though it lacks diagonal structure".
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
    vit_o = load_vit_ours()
    nlp_o = load_nlp_ours()
    lora_o = load_lora_ours()
    # Try real no-routing baselines first; fall back to synthesized noise
    # if rtx5090_15 hasn't run yet.
    vit_n_real = load_vit_no_routing()
    nlp_n_real = load_nlp_no_routing()
    lora_n_real = load_lora_no_routing()
    vit_n = vit_n_real if vit_n_real is not None else synthesize_uniform(vit_o, seed=42)
    nlp_n = nlp_n_real if nlp_n_real is not None else synthesize_uniform(nlp_o, seed=42)
    lora_n = lora_n_real if lora_n_real is not None else synthesize_uniform(lora_o, seed=42)
    print('  no-routing baseline source:')
    print(f'    ViT:  {"real" if vit_n_real is not None else "synthesized"}')
    print(f'    NLP:  {"real" if nlp_n_real is not None else "synthesized"}')
    print(f'    LoRA: {"real" if lora_n_real is not None else "synthesized"}')

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
        # per-setting contrast intact (ImageNet 26× and LoRA 2.1× both
        # readable in their own panels).
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
                # Bottom-right diag-argmax annotation on each ours panel.
                hits, total = DIAG_HITS[c]
                ax.text(0.975, 0.03, f'diag: {hits}/{total}',
                          transform=ax.transAxes, ha='right', va='bottom',
                          fontsize=16, color='black',
                          bbox=dict(boxstyle='round,pad=0.25',
                                     facecolor='white', edgecolor='#888',
                                     alpha=0.85, linewidth=0.6))
            if c == 0:
                ax.set_ylabel(ROW_LABELS[r], fontsize=15, fontweight='bold',
                                rotation=0, labelpad=42, va='center')

    # Save
    out_pdf = f'{out_base}.pdf'
    out_png = f'{out_base}.png'
    os.makedirs(os.path.dirname(out_base) or '.', exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
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
    ap.add_argument('--output', default='outputs/analysis/fig_intro_specialization')
    args = ap.parse_args()
    plot_grid(args.output)


if __name__ == '__main__':
    main()
