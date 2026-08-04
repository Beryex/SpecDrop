#!/bin/bash
###############################################################################
# RTX 5090 Script 3t: NLP mini-ablation — Phase T (linear warmup schedule
# test at Phase P root: uniform + SE=1.0 + per-cat + wr=1.0, pa ∈ {0.6, 0.7}).
#
# Motivation. Phase D and Phase P are our two primary ours-candidates:
#   Phase D (uniform + SE=1.0 + scalar + wr=0):       pa=0.6 → 55.15 ★
#   Phase P (uniform + SE=1.0 + per-cat + wr=1.0):    pa=0.6 → 55.17
# The original Phase S tested linear warmup at Phase F root (uniform +
# scalar + wr=1.0 + NO SE), which is the wrong reference — our actual
# ours configs all have SE=1.0. Phase T tests linear at Phase P root to
# answer: "does linear warmup shift BEST_PA or improve PPL in the config
# we actually care about?"
#
# At pa=0.5 under wr=1.0 the schedule is mathematically a no-op (mask
# stays at S/K regardless of cosine/linear), so pa=0.5 is omitted.
# pa=0.6 is our current BEST_PA; pa=0.7 is where schedule differences
# are most visible (middle of the warmup-matters range).
#
# Search: pa ∈ {0.6, 0.7} with linear warmup (3 seeds each)
# Fixed:  uniform K=7, ffn_per_branch=192, SE_dim=192 (from
#         split_ffn_budget_for_se(1540, 1.0, 7)), per-cat routing
#         with β=1.0, wr=1.0, 100M tokens, 10 epochs
# Compare to Phase P (same config, cosine warmup):
#   pa=0.6 cosine = 55.17 ± 0.19
#   pa=0.7 cosine = 55.56 ± 0.20
#
# 2 pa × 3 seeds = 6 runs, 2 per GPU seed, ~2.4h wall-clock across 3 GPUs.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseT_pa{pa}_wr1.0_linear_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase T — linear warmup @ Phase P root"
echo " (uniform + SE=1.0 + per-cat + wr=1.0, pa ∈ {0.6, 0.7}, LINEAR)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local SEED=$1 PA=$2 PI=$3
    local ENAME="phaseT_pa${PA}_wr1.0_linear_s${SEED}"
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

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[cfg-gen] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}'
      f'  total={7*ffn_per_branch + se_dim}  (target 1540)')

fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

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
                   'frac_per_category': fracs, 'amplification_beta': 1.0,
                   'warmup_schedule': 'linear'},
    'training': {'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
                  'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
                  'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'},
    'data': {'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
              'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
print(f'[cfg-gen] Phase T: uniform + SE=1.0 + per-cat + wr=1.0 LINEAR, pa={PA} pi={PI}')
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    run_cell "$SEED" "0.6" "0.4"
    run_cell "$SEED" "0.7" "0.3"
done

echo ""
echo "============================================================"
echo " Phase T summary (linear warmup at Phase P root, pa ∈ {0.6, 0.7})"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
# Phase P (cosine) reference for same config
cosine_ref = {'0.6': 55.17, '0.7': 55.56}
print(f"{'pa':>4}  {'T linear':>14}  {'P cosine':>14}  {'Δ (T−P)':>9}")
for pa in ['0.6', '0.7']:
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseT_pa{pa}_wr1.0_linear_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((v-m)**2 for v in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
        delta = m - cosine_ref[pa]
        print(f"{pa:>4}  {m:>6.2f} ± {std:>4.2f}  {cosine_ref[pa]:>14.2f}  {delta:>+9.2f}")

print()
print("Reference configs:")
print("  Phase D pa=0.5 (scalar, wr=0, SE=1.0) = 55.18  ← Phase D anchor")
print("  Phase D pa=0.6 (scalar, wr=0, SE=1.0) = 55.15  ← current BEST overall")
print("  Phase P pa=0.5 (per-cat, wr=1.0 cos, SE=1.0) = 55.17")
print("  Phase P pa=0.6 (per-cat, wr=1.0 cos, SE=1.0) = 55.17")
print()
print("Interpretation:")
print("  If Phase T pa=0.6 < 55.15 → linear warmup beats Phase D anchor → new ours")
print("  If Phase T pa=0.6 ≈ 55.17 → schedule shape doesn't matter at Phase P root")
print("  If Phase T pa=0.7 < 55.17 → linear shifts BEST_PA to 0.7")
EOF
echo "Done: $(date)"
