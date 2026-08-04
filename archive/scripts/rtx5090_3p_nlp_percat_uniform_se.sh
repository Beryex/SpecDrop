#!/bin/bash
###############################################################################
# RTX 5090 Script 3p: NLP mini-ablation — Phase P (two-mechanism combo:
# UNIFORM branches + PER-CATEGORY routing + SE=1.0x, pa sweep at wr=1.0).
#
# Motivation. Phase O tests the kitchen-sink (non-uniform + per-cat + SE).
# Phase P is the companion ablation: drop the non-uniform ingredient, keep
# SE + per-cat, to isolate whether the routing-curve-flattening effect
# from SE + per-cat is enough on its own to shift BEST_PA > 0.5, or
# whether non-uniform's non-additive synergy (Phase N's gap-flip was
# driven mostly by the non-uniform × per-cat interaction) is essential.
#
# Decomposition matrix (100M mini-ablation, pa sweep at wr=1.0):
#     Phase F  — uniform  scalar         no SE
#     Phase M  — uniform  per-cat        no SE
#     Phase D  — uniform  scalar         SE=1.0   (pa=0.6 vs 0.5: -0.03)
#     Phase P  — uniform  per-cat        SE=1.0   (THIS SCRIPT)
#     Phase J  — non-uni  scalar         no SE
#     Phase N  — non-uni  per-cat        no SE    (pa=0.6 vs 0.5: -0.06)
#     Phase L  — non-uni  scalar         SE=1.0   (not run yet)
#     Phase O  — non-uni  per-cat        SE=1.0   (kitchen sink)
#
# Budget (SE=1.0x, uniform): total 1540 splits as branch_total=1348 + SE=192.
# Uniform branches: ffn_dim_per_branch = 1348 / 7 ≈ 192 per branch.
# Total params = 7*192 + 192 = 1536 ≈ dense 1536.
#
# Search: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa, wr=1.0 fixed,
#         SE_ratio=1.0 fixed, uniform branches, per-cat routing (β=1.0).
# 6 pa × 3 seeds = 18 runs, ~5h wall-clock on 3 GPUs.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseP_pa{pa}_wr1.0_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase P — uniform branches + per-cat + SE=1.0x"
echo " pa sweep @ wr=1.0"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseP_pa${PA}_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, DOMAIN_NAMES)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] cache: {cache}')

# Split dense-equivalent budget between branches and SE, then divide the
# branch portion uniformly across K branches.
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7  # uniform, truncate to int
print(f'[cfg-gen] SE_ratio={SE_RATIO}  total_branch_ffn={total_branch_ffn}  SE_dim={se_dim}')
print(f'[cfg-gen] ffn_per_branch (uniform) = {ffn_per_branch}')
print(f'[cfg-gen] total params budget: K * ffn_per_branch + SE_dim = '
      f'{7*ffn_per_branch + se_dim} (target 1540)')

fracs = compute_category_fractions(cache, num_categories=7)
print(f'[cfg-gen] per-category fractions (pa={PA}, wr=1.0, SE={SE_RATIO}x):')
for c, name in enumerate(DOMAIN_NAMES):
    print(f"  c{c}: {name:<30} {fracs[c]:>7.2%}")
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
assert max(range(7), key=lambda i: fracs[i]) == 0, 'CC (id=0) should have max frac'

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': ffn_per_branch,   # scalar → uniform
        'shared_expert_dim': se_dim,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': 1.0},
    'training': {'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
                  'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
                  'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'},
    'data': {'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
              'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase P] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase P summary (uniform + per-cat + SE=1.0x, pa sweep)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'pa':>5} {'pi':>5}  {'mean PPL':>10}  {'std':>6}  n")
rows = []
for pa in ['0.5','0.6','0.7','0.8','0.9','1.0']:
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseP_pa{pa}_wr1.0_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
complete = [r for r in rows if r[4] == 3]
if complete:
    best = min(complete, key=lambda r: r[2])
    print(f"\nBEST_PA_P (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
    print(f"Reference (100M tokens):")
    print(f"  Phase D (uniform + SE=1.0, scalar)  BEST_PA=0.5 → 55.18")
    print(f"  Phase M (uniform + per-cat, no SE)  BEST_PA=0.5 → 56.80")
    print(f"  Phase N (non-uni + per-cat, no SE)  BEST_PA=0.6 → 55.40")
    print(f"  Phase O (non-uni + per-cat + SE)    running in parallel")
EOF
echo "Done: $(date)"
