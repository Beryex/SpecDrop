#!/bin/bash
###############################################################################
# RTX 5090 Script 3v: NLP mini-ablation — Phase V (amplification_beta sweep at
# Phase P root: uniform + SE=1.0 + per-cat + wr=1.0 COSINE, pa ∈ {0.6, 0.7}).
#
# Motivation. β controls the asymmetry strength in per-category routing:
#     gap_c = g · [(1 - frac_c) / (1 - 1/M)]^β
# β=1 (our default so far) is linear. β>1 amplifies the asymmetry: large
# domains (CC, C4) get even more ensemble-like weights, small domains
# (Wiki, SE) get even stronger routing. Phase V tests whether tuning β can
# improve over Phase P β=1.
#
# - pa=0.6 is our current BEST_PA. β effect expected small here (g=0.2 small).
# - pa=0.7 is where per-cat rescue is strongest (Phase P saves 0.06 vs Phase D,
#   Phase T linear saves another 0.13). If β>1 amplifies, pa=0.7 could drop
#   below 55.2 and shift BEST_PA up to 0.7.
#
# Search: pa ∈ {0.6, 0.7} × β ∈ {2.0, 4.0}, 3 seeds each
# Fixed:  uniform K=7, ffn=192 + SE=192, per-cat, wr=1.0 COSINE, 100M, 10ep
# Reference (β=1.0):
#   Phase P pa=0.6 β=1.0 = 55.17 ± 0.19
#   Phase P pa=0.7 β=1.0 = 55.56 ± 0.20
# Also:
#   Phase D pa=0.6 (scalar, wr=0, SE=1.0) = 55.15  ← overall BEST anchor
#
# Constraint check: max_gap = min(S, (K-S)/(K-1)).
#   pa=0.6 (S=3.0): max_gap=0.667; SE gap at β=4: 0.317 ✓
#   pa=0.7 (S=2.5): max_gap=0.750; SE gap at β=4: 0.634 ✓
# Both safe from clamp.
#
# 2 pa × 2 β × 3 seeds = 12 runs, 4 per GPU seed, ~4.8h wall-clock across 3 GPUs.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseV_pa{pa}_wr1.0_beta{β}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase V — β sweep @ Phase P root"
echo " (uniform + SE=1.0 + per-cat + wr=1.0 cosine, pa × β sweep)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local SEED=$1 PA=$2 PI=$3 BETA=$4
    local ENAME="phaseV_pa${PA}_wr1.0_beta${BETA}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI BETA=$BETA $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, DOMAIN_NAMES)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] cache: {cache}  β={BETA}')

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[cfg-gen] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}'
      f'  total={7*ffn_per_branch + se_dim}')

fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

# Show what per-cat looks like under this β
S = PA + 6 * PI
SK = S / 7
g = PA - PI
denom = 1 - 1/7
print(f'[cfg-gen] β={BETA}  pa={PA}  pi={PI}  S={S:.3f}  S/K={SK:.4f}  g={g:.3f}')
print(f'[cfg-gen] per-cat targets (domain_id order):')
for c, name in enumerate(DOMAIN_NAMES):
    r = (1 - fracs[c]) / denom
    gap = g * (r ** BETA)
    pac = SK + gap * 6/7
    pic = SK - gap / 7
    print(f"  c{c}: {name:<26} frac={fracs[c]:>7.2%}  r^β={r**BETA:>6.3f}  "
          f"gap={gap:>6.3f}  p_a^c={pac:>6.3f}  p_i^c={pic:>6.3f}")

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': ffn_per_branch,
        'shared_expert_dim': se_dim,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': BETA,
                   'warmup_schedule': 'cosine'},
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
    echo "[Phase V] seed=$SEED"
    for PA in 0.6 0.7; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        for BETA in 2.0 4.0; do
            run_cell "$SEED" "$PA" "$PI" "$BETA"
        done
    done
done

echo ""
echo "============================================================"
echo " Phase V summary (β sweep at Phase P root)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
# Phase P (β=1) reference
ref = {'0.6': 55.17, '0.7': 55.56}
print(f"{'pa':>4}  {'β':>5}  {'mean PPL':>12}  {'std':>5}  Δ vs β=1.0")
rows = {}
for pa in ['0.6', '0.7']:
    for beta in ['2.0', '4.0']:
        ppls = []
        for s in (42, 123, 456):
            p = os.path.join(base, f'phaseV_pa{pa}_wr1.0_beta{beta}_s{s}', 'results.json')
            if os.path.exists(p):
                ppls.append(json.load(open(p))['best_val_ppl'])
        if ppls:
            m = sum(ppls)/len(ppls); std = (sum((v-m)**2 for v in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
            delta = m - ref[pa]
            rows[(pa, beta)] = m
            print(f"{pa:>4}  {beta:>5}  {m:>7.2f} ± {std:>4.2f}  {delta:>+10.2f}")
    # Print the β=1 baseline row inline
    print(f"{pa:>4}  {'1.0':>5}  {ref[pa]:>7.2f}  {'':>10}  {0.0:>+10.2f}  ← Phase P baseline")
    print()

# BEST_PA shift check
if rows:
    print("BEST_PA shift check:")
    for beta in ['1.0', '2.0', '4.0']:
        ppl_06 = rows.get(('0.6', beta), ref['0.6'] if beta == '1.0' else None)
        ppl_07 = rows.get(('0.7', beta), ref['0.7'] if beta == '1.0' else None)
        if ppl_06 is not None and ppl_07 is not None:
            winner = 'pa=0.6' if ppl_06 < ppl_07 else 'pa=0.7'
            print(f"  β={beta}: pa=0.6→{ppl_06:.2f}, pa=0.7→{ppl_07:.2f}, winner={winner} (Δ={abs(ppl_06-ppl_07):.2f})")

# Reference anchor
print()
print("Anchor: Phase D pa=0.6 (scalar, wr=0, SE=1.0) = 55.15  ← overall BEST in mini-ablation")
EOF
echo "Done: $(date)"
