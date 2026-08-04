#!/bin/bash
###############################################################################
# RTX 5090 Script 3q: NLP mini-ablation — Phase Q (SE ratio sweep at Phase D
# root: uniform branches + scalar routing + wr=0, pa ∈ {0.5, 0.6}).
#
# Motivation. Phase D (uniform + SE=1.0 + scalar + wr=0) is currently the
# lowest-PPL cell in mini-ablation (pa=0.6 → 55.15). But Phase D only tested
# SE_ratio=1.0 — we have no data for SE=2.0x or 4.0x with this root config.
# Phase Q fills this gap.
#
# Search: SE_ratio ∈ {2.0x, 4.0x} × pa ∈ {0.5, 0.6}
# Fixed:  uniform branches, scalar routing, wr=0, K=7, 100M tokens, 10 epochs
# Budget: total 1540 splits as:
#     SE=2.0x → branch_total=1222, SE=356 → uniform ffn=174, SE=356
#     SE=4.0x → branch_total=980,  SE=560 → uniform ffn=140, SE=560
#
# 2 SE × 2 pa × 3 seeds = 12 runs, ~4.8h on 3 GPUs (4 runs per seed).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseQ_se{X}x_pa{pa}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase Q — SE ratio sweep @ Phase D root"
echo " (uniform + scalar + wr=0, pa ∈ {0.5, 0.6}, SE ∈ {2.0x, 4.0x})"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local X=$1 PA=$2 PI=$3 SEED=$4
    local ENAME="phaseQ_se${X}x_pa${PA}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI SE_RATIO=$X $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_ffn_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = float(os.environ['SE_RATIO'])

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7  # uniform truncation
print(f'[cfg-gen] SE_ratio={SE_RATIO}  branch_total={total_branch_ffn}  SE_dim={se_dim}')
print(f'[cfg-gen] ffn_per_branch={ffn_per_branch}  (uniform)')
print(f'[cfg-gen] budget check: K*ffn + SE = {7*ffn_per_branch + se_dim}  (target 1540)')

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
                   'assignment': 'round_robin', 'warmup_ratio': 0.0},
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
    echo "[Phase Q] seed=$SEED"
    for X in 2.0 4.0; do
        for PA in 0.5 0.6; do
            PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
            run_cell "$X" "$PA" "$PI" "$SEED"
        done
    done
done

echo ""
echo "============================================================"
echo " Phase Q summary (SE sweep at Phase D root)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'SE':>6}  {'pa':>4}  {'mean PPL':>10}  {'std':>6}  n")
rows = []
for x in ['2.0', '4.0']:
    for pa in ['0.5', '0.6']:
        ppls = []
        for s in (42, 123, 456):
            p = os.path.join(base, f'phaseQ_se{x}x_pa{pa}_s{s}', 'results.json')
            if os.path.exists(p):
                ppls.append(json.load(open(p))['best_val_ppl'])
        if ppls:
            m = sum(ppls)/len(ppls); std = (sum((v-m)**2 for v in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
            rows.append((x, pa, m, std, len(ppls)))
            print(f"{x+'x':>6}  {pa:>4}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
print()
print("Reference (uniform + scalar + wr=0 cells):")
print("  Phase D SE=1.0 pa=0.5 → 55.18")
print("  Phase D SE=1.0 pa=0.6 → 55.15 ★ current best")
print("  Phase A SE=0.0 pa=0.5 → 56.74")
print("  Phase A SE=0.0 pa=0.6 → 56.78")
EOF
echo "Done: $(date)"
