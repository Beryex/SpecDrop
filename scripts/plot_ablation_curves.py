"""4×3 ablation grid: 4 settings (CIFAR / ViT / NLP / LoRA) × 3 sweeps
(p_a / row-2 / SE).

Row-2 axis differs by setting:
- CIFAR: w_r (warmup ratio) — Phase B
- ViT / NLP / LoRA: β (per-cat amplification) — phase{5,3,8}b

Each panel plots mean ± std across 3 seeds (s42, s123, s456), with a
vertical dashed line at the deployed operating point.
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.rcParams.update({'pdf.fonttype': 42,
                            'xtick.labelsize': 13.5, 'ytick.labelsize': 13.5})
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)

SEEDS = (42, 123, 456)


# ─── Metric reader ────────────────────────────────────────────────────────
def _read(run_dir: str, key: str):
    p = f'{run_dir}/results.json'
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get(key)
    except Exception:
        return None


def _aggregate(seed_to_candidates: dict, metric_key: str):
    """seed_to_candidates: {seed: [glob_pattern, ...]} (first match wins).
    Returns (mean, std, n, raw_values) or None if no seed yielded a value."""
    vals = []
    for seed in SEEDS:
        candidates = seed_to_candidates.get(seed, [])
        for g in candidates:
            for d in sorted(glob.glob(g)):
                v = _read(d, metric_key)
                if v is not None:
                    vals.append(float(v))
                    break
            else:
                continue
            break
    if not vals:
        return None
    arr = np.array(vals)
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return arr.mean(), std, len(arr), arr.tolist()


# ─── Per-setting glob factories ───────────────────────────────────────────
# CIFAR: rtx5090_ablation/ablation_pa{X}_pi{1-X}_s{seed}
#        rtx5090_ablation/ablation_wr{X}_pa0.7_s{seed}
#        rtx5090_ablation/ablation_se_ratio_{X}x_pa0.7_wr1.0_s{seed}
def cifar_pa(pa, seed):
    return [f'outputs/rtx5090_ablation/ablation_pa{pa}_pi*_s{seed}']


def cifar_wr(wr, seed):
    return [f'outputs/rtx5090_ablation/ablation_wr{wr}_pa0.7_s{seed}']


def cifar_se(se, seed):
    return [f'outputs/rtx5090_ablation/ablation_se_ratio_{se}x_pa0.7_wr1.0_s{seed}']


# ViT: phase5a_pa{X}_se1.0 / phase5b_pa0.6_beta{X}_se1.0 / phase5c_pa0.6_beta1.0_se{X}
# β=1 reuses 5a pa=0.6; SE=1.0 reuses 5a/5b anchor.
def vit_pa(pa, seed):
    return [f'outputs/rtx5090_vit_mini_ablation/phase5a_pa{pa}_se1.0_s{seed}']


def vit_beta(b, seed):
    if b == 1.0:
        return [f'outputs/rtx5090_vit_mini_ablation/phase5a_pa0.6_se1.0_s{seed}']
    bs = '0' if b == 0 else f'{b:.1f}'
    return [f'outputs/rtx5090_vit_mini_ablation/phase5b_pa0.6_beta{bs}_se1.0_s{seed}']


def vit_se(se, seed):
    if se == 1.0:
        return [f'outputs/rtx5090_vit_mini_ablation/phase5a_pa0.6_se1.0_s{seed}']
    ss = '0' if se == 0 else f'{se:.1f}'
    return [f'outputs/rtx5090_vit_mini_ablation/phase5c_pa0.6_beta1.0_se{ss}_s{seed}']


# NLP: phase3a_pa{X}_se1.0 / phase3b_pa0.6_beta{X}_se1.0 / phase3c_pa0.6_beta4.0_se{X}
# β=1 reuses 3a pa=0.6; SE=1.0 reuses 3b pa=0.6 β=4 dir for SE sweep anchor.
def nlp_pa(pa, seed):
    return [f'outputs/rtx5090_nlp_mini_ablation/phase3a_pa{pa}_se1.0_s{seed}']


def nlp_beta(b, seed):
    if b == 1.0:
        return [f'outputs/rtx5090_nlp_mini_ablation/phase3a_pa0.6_se1.0_s{seed}']
    bs = '0' if b == 0 else f'{b:.1f}'
    return [f'outputs/rtx5090_nlp_mini_ablation/phase3b_pa0.6_beta{bs}_se1.0_s{seed}']


def nlp_se(se, seed):
    if se == 1.0:
        return [f'outputs/rtx5090_nlp_mini_ablation/phase3b_pa0.6_beta4.0_se1.0_s{seed}']
    ss = '0' if se == 0 else f'{se:.1f}'
    return [f'outputs/rtx5090_nlp_mini_ablation/phase3c_pa0.6_beta4.0_se{ss}_s{seed}']


# LoRA: phase8a_pa{X}_se1.0 / phase8b_pa0.8_beta{X}_se1.0 / phase8c_pa0.8_beta1.0_se{X}
def lora_pa(pa, seed):
    return [f'outputs/rtx5090_lora_ablation/phase8a_pa{pa}_se1.0_s{seed}']


def lora_beta(b, seed):
    if b == 1.0:
        return [f'outputs/rtx5090_lora_ablation/phase8a_pa0.8_se1.0_s{seed}']
    bs = '0' if b == 0 else f'{b:.1f}'
    return [f'outputs/rtx5090_lora_ablation/phase8b_pa0.8_beta{bs}_se1.0_s{seed}']


def lora_se(se, seed):
    if se == 1.0:
        return [f'outputs/rtx5090_lora_ablation/phase8a_pa0.8_se1.0_s{seed}']
    ss = '0' if se == 0 else f'{se:.1f}'
    return [f'outputs/rtx5090_lora_ablation/phase8c_pa0.8_beta1.0_se{ss}_s{seed}']


# ─── Panel registry ───────────────────────────────────────────────────────
# (row_setting, col_sweep) → dict with: xs, factory, metric, higher_is_better,
# deployed_x, xlabel, ylabel, ylim_padding
PANELS = [
    # row 0 — CIFAR-100
    {
        'setting': 'CIFAR-100', 'sweep': 'pa',
        'xs':       [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        'factory':  cifar_pa, 'metric': 'best_top1', 'hib': True,
        'deployed': 0.7, 'xlabel': r'$p_a$', 'ylabel': 'Top-1 (%)',
    },
    {
        'setting': 'CIFAR-100', 'sweep': 'wr',
        'xs':       [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
        'factory':  cifar_wr, 'metric': 'best_top1', 'hib': True,
        'deployed': 1.0, 'xlabel': r'$w_r$', 'ylabel': 'Top-1 (%)',
    },
    {
        'setting': 'CIFAR-100', 'sweep': 'se',
        'xs':       [0, 0.25, 0.5, 1.0, 2.0, 4.0],
        'factory':  cifar_se, 'metric': 'best_top1', 'hib': True,
        'deployed': 0, 'xlabel': r'SE ratio ($\times$)', 'ylabel': 'Top-1 (%)',
    },
    # row 1 — ViT
    {
        'setting': 'ImageNet ViT', 'sweep': 'pa',
        'xs':       [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'factory':  vit_pa, 'metric': 'best_top1', 'hib': True,
        'deployed': 0.6, 'xlabel': r'$p_a$', 'ylabel': 'Top-1 (%)',
    },
    {
        'setting': 'ImageNet ViT', 'sweep': 'beta',
        'xs':       [0, 1.0, 2.0, 4.0],
        'factory':  vit_beta, 'metric': 'best_top1', 'hib': True,
        'deployed': 1.0, 'xlabel': r'$\beta$', 'ylabel': 'Top-1 (%)',
    },
    {
        'setting': 'ImageNet ViT', 'sweep': 'se',
        'xs':       [0, 0.5, 1.0, 2.0],
        'factory':  vit_se, 'metric': 'best_top1', 'hib': True,
        'deployed': 2.0, 'xlabel': r'SE ratio ($\times$)', 'ylabel': 'Top-1 (%)',
    },
    # row 2 — NLP SlimPajama
    {
        'setting': 'SlimPajama', 'sweep': 'pa',
        'xs':       [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'factory':  nlp_pa, 'metric': 'best_val_ppl', 'hib': False,
        'deployed': 0.6, 'xlabel': r'$p_a$', 'ylabel': 'Val PPL',
    },
    {
        'setting': 'SlimPajama', 'sweep': 'beta',
        'xs':       [0, 1.0, 2.0, 4.0],
        'factory':  nlp_beta, 'metric': 'best_val_ppl', 'hib': False,
        'deployed': 4.0, 'xlabel': r'$\beta$', 'ylabel': 'Val PPL',
    },
    {
        'setting': 'SlimPajama', 'sweep': 'se',
        'xs':       [0, 0.5, 1.0, 2.0],
        'factory':  nlp_se, 'metric': 'best_val_ppl', 'hib': False,
        'deployed': 0.5, 'xlabel': r'SE ratio ($\times$)', 'ylabel': 'Val PPL',
    },
    # row 3 — LoRA SuperNI
    {
        'setting': 'LoRA SuperNI', 'sweep': 'pa',
        'xs':       [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'factory':  lora_pa, 'metric': 'eval_rouge_l', 'hib': True,
        'deployed': 0.8, 'xlabel': r'$p_a$', 'ylabel': 'ROUGE-L',
    },
    {
        'setting': 'LoRA SuperNI', 'sweep': 'beta',
        'xs':       [0, 1.0, 2.0, 4.0],
        'factory':  lora_beta, 'metric': 'eval_rouge_l', 'hib': True,
        'deployed': 1.0, 'xlabel': r'$\beta$', 'ylabel': 'ROUGE-L',
    },
    {
        'setting': 'LoRA SuperNI', 'sweep': 'se',
        'xs':       [0, 0.5, 1.0, 2.0],
        'factory':  lora_se, 'metric': 'eval_rouge_l', 'hib': True,
        'deployed': 1.0, 'xlabel': r'SE ratio ($\times$)', 'ylabel': 'ROUGE-L',
    },
]


# ─── Main ──────────────────────────────────────────────────────────────────
def main(out_base: str = 'outputs/analysis/fig_ablation_curves'):
    # 3 rows (pa / row-2 / SE) × 4 cols (CIFAR / ViT / NLP / LoRA).
    # Each column = one setting (own y-metric); each row = one sweep type
    # (own x-axis variable).
    fig, axes = plt.subplots(3, 4, figsize=(10.0, 4.35),
                             gridspec_kw={'wspace': 0.32, 'hspace': 0.55,
                                          'left': 0.055, 'right': 0.99,
                                          'top': 0.91, 'bottom': 0.10})

    line_color = '#5B7FBF'   # paper blue
    err_color  = '#5B7FBF'
    deploy_color = '#C0504D'  # subtle red for deployed marker
    chain_color  = '#666666'  # grey for "previous-row best" reference

    settings = ['CIFAR-100', 'ImageNet ViT', 'SlimPajama', 'LoRA SuperNI']
    sweep_to_row = {'pa': 0, 'wr': 1, 'beta': 1, 'se': 2}

    # ─── Pass 1: aggregate every panel, cache (xs, means, stds) ────────────
    print('Aggregating ablation cells...')
    print('=' * 70)
    panel_data = {}  # (setting, sweep) -> (xs_list, means_list, stds_list)
    for panel in PANELS:
        sweep = panel['sweep']
        xs, ys, ystd = [], [], []
        per_seed_print = []
        for x in panel['xs']:
            seed_to_candidates = {s: panel['factory'](x, s) for s in SEEDS}
            agg = _aggregate(seed_to_candidates, panel['metric'])
            if agg is None:
                per_seed_print.append(f'  {sweep}={x}: MISSING')
                continue
            mean, std, n, raw = agg
            xs.append(float(x)); ys.append(mean); ystd.append(std)
            per_seed_print.append(
                f'  {sweep}={x}: n={n}  {panel["metric"]}={mean:.3f}±{std:.3f}'
                f'  (raw={[round(v,3) for v in raw]})'
            )
        print(f'[{panel["setting"]} / {sweep}]')
        for line in per_seed_print:
            print(line)
        panel_data[(panel['setting'], sweep)] = (xs, ys, ystd)

    # ─── Look up "previous-row deployed value" for chain reference lines ───
    def _prev_row_anchor(setting: str, current_sweep: str):
        """Returns (sweep_name, value) for the row above, or None."""
        if current_sweep == 'pa':
            return None
        # Row 1 (wr/beta) → ref is row 0 (pa) at deployed pa
        if current_sweep in ('wr', 'beta'):
            pa_panel = next(p for p in PANELS
                             if p['setting'] == setting and p['sweep'] == 'pa')
            xs, ys, _ = panel_data[(setting, 'pa')]
            if pa_panel['deployed'] not in xs:
                return None
            return ('pa', ys[xs.index(pa_panel['deployed'])])
        # Row 2 (se) → ref is row 1 (wr or beta) at its deployed value
        if current_sweep == 'se':
            for r1_sweep in ('wr', 'beta'):
                r1_panel = next((p for p in PANELS
                                  if p['setting'] == setting
                                  and p['sweep'] == r1_sweep), None)
                if r1_panel is None:
                    continue
                xs, ys, _ = panel_data[(setting, r1_sweep)]
                if r1_panel['deployed'] not in xs:
                    return None
                return (r1_sweep, ys[xs.index(r1_panel['deployed'])])
        return None

    # ─── Pass 2: render every panel ─────────────────────────────────────────
    for panel in PANELS:
        col = settings.index(panel['setting'])
        sweep = panel['sweep']
        row = sweep_to_row[sweep]
        ax = axes[row, col]
        xs, ys, ystd = panel_data[(panel['setting'], sweep)]

        if not xs:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=15, color='#888')
            ax.set_xlabel(panel['xlabel'], fontsize=15)
            continue

        xs_a = np.array(xs); ys_a = np.array(ys); ystd_a = np.array(ystd)

        # Chain reference line: previous row's deployed value (rows 1, 2 only).
        anchor = _prev_row_anchor(panel['setting'], sweep)
        if anchor is not None:
            ref_label_map = {'pa': r'$p_a$', 'wr': r'$w_r$', 'beta': r'$\beta$'}
            ax.axhline(anchor[1], color=chain_color, linestyle=':',
                        linewidth=1.0, alpha=0.85, zorder=1,
                        label=f'best @ {ref_label_map[anchor[0]]} sweep')

        # Vertical dashed line at deployed value of THIS sweep.
        ax.axvline(panel['deployed'], color=deploy_color, linestyle='--',
                   linewidth=1.0, alpha=0.7, zorder=1)

        # Error-bar curve.
        ax.errorbar(xs_a, ys_a, yerr=ystd_a, color=line_color, ecolor=err_color,
                    marker='o', markersize=5.5, linewidth=1.6, capsize=3,
                    elinewidth=1.0, zorder=3)

        # Highlight deployed point.
        if panel['deployed'] in xs:
            i = xs.index(panel['deployed'])
            ax.scatter([xs_a[i]], [ys_a[i]], s=80, facecolor=deploy_color,
                       edgecolor='white', linewidth=1.3, zorder=4)

        # Mech-OFF marker on p_a sweeps: pa=0.5 is the algebraic mechanism-
        # OFF point (uniform 1/K merge), a priori excluded from optimum
        # search per paper §5.5. Render as a hollow grey marker that
        # overlays the errorbar's solid blue dot, making the "this point
        # is structurally not in the search space" status visually obvious
        # and preempting "why didn't you pick the lower/higher one?" reader
        # questions on NLP + ViT (where pa=0.5 looks competitive).
        if sweep == 'pa' and 0.5 in xs:
            i_off = xs.index(0.5)
            ax.scatter([xs_a[i_off]], [ys_a[i_off]], s=70,
                        facecolor='white', edgecolor='#888',
                        linewidth=1.4, zorder=3.5,
                        label=r'$p_a{=}0.5$ (mech-OFF)')

        # Tighten y-axis: cover (mean ± std) range AND chain ref line, with
        # ~15% padding so errorbar caps + nearest tick are visibly inside
        # the panel (default 10% left some panels with caps grazing top).
        lo_data = float(np.min(ys_a - ystd_a))
        hi_data = float(np.max(ys_a + ystd_a))
        if anchor is not None:
            lo_data = min(lo_data, anchor[1])
            hi_data = max(hi_data, anchor[1])
        span = max(hi_data - lo_data, 1e-9)
        pad  = span * 0.15
        ax.set_ylim(lo_data - pad, hi_data + pad)

        # Denser y-ticks: ~6 evenly-spaced major ticks per panel, snapped to
        # nice numbers. NO prune — keep edge ticks visible so readers can
        # read the full data envelope (prune='both' was hiding endpoint
        # ticks, making errorbar tips appear to stick out past visible
        # range even though they were inside ylim).
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))

        # Axes.
        ax.set_xlabel(panel['xlabel'], fontsize=15)
        ax.set_ylabel(panel['ylabel'], fontsize=14, labelpad=4)
        if row == 0:
            ax.set_title(panel['setting'], fontsize=16, pad=8,
                          fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=9)
        for s in ax.spines.values():
            s.set_color('#666'); s.set_linewidth(0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#bbb', alpha=0.6)
        if anchor is not None:
            ax.legend(loc='best', fontsize=11.5, frameon=False, handlelength=1.5,
                       borderaxespad=0.2)

    out_pdf = f'{out_base}.pdf'
    out_png = f'{out_base}.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    print('=' * 70)
    print(f'wrote {out_pdf}')
    print(f'wrote {out_png}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/analysis/fig_ablation_curves')
