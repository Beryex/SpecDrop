#!/bin/bash
###############################################################################
# RTX 5090 Script 3d: NLP mini-ablation — per-STEP pa warmup variant of the
# Phase P lock-in config (pa=0.6, β=2, SE_ratio=0.5, per-cat).
#
# Motivation. trainer_nlp.py schedules LR per-step (cosine decay after linear
# warmup_steps=1000 over the full 30k+ step NLP budget). But SoftSpecDrop's pa
# warmup is DRIVEN BY current_epoch, updated once per epoch via on_epoch_end.
# At 100M × 10 epochs, that's only 10 distinct pa-state changes across 30k
# optimizer steps — a coarse staircase vs. the LR's smooth curve.
#
# This script tests whether switching pa warmup to per-step granularity
# (warmup_unit='step' added in this session) changes 3-seed PPL meaningfully
# at the Phase P lock-in config.
#
# Config (matches the 500M ours config [9] in rtx5090_4, scaled to 100M):
#     pa = 0.6, pi = 0.4, SE_ratio = 0.5 (ffn=205, SE_dim=103)
#     per-cat β = 2.0 (strict argmin from 3b)
#     wr = 1.0, warmup_schedule = cosine
#     warmup_unit = 'step'   ← ONLY difference vs the 100M baseline
#     frac_per_category computed from 100M train cache at config-gen
#
# Baseline for comparison: phase3c_pa0.6_beta2.0_se0.5_s{42,123,456}
#   3-seed mean = 55.07 PPL (epoch-based warmup).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phase3d_pa0.6_beta2.0_se0.5_stepwarmup_s{seed}/
# Per GPU: 1 run × ~70 min. Wall clock ≈ 70 min on 3 GPUs in parallel.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " 3d — per-STEP pa warmup @ Phase P lock-in (100M)"
echo " (pa=0.6 β=2 SE=0.5 per-cat wr=1.0 cosine, warmup_unit=step)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local SEED=$1
    local WUNIT=${2:-step}
    local ENAME
    if [ "$WUNIT" = "step" ]; then
        ENAME="phase3d_pa0.6_beta2.0_se0.5_stepwarmup_s${SEED}"
    else
        ENAME="phase3c_pa0.6_beta2.0_se0.5_s${SEED}"
    fi
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED WUNIT=$WUNIT $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED'])
PA = 0.6; PI = 0.4; BETA = 2.0; SE_RATIO = 0.5

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

print(f'[cfg-gen] pa={PA} β={BETA} SE_ratio={SE_RATIO} warmup_unit=step')
print(f'[cfg-gen] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={7*ffn_per_branch + se_dim}')

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
                   'warmup_schedule': 'cosine',
                   'warmup_unit': os.environ.get('WUNIT', 'step')},
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
    # epoch-warmup baseline cell (generated here so the comparison below is
    # self-contained; skipped when already present)
    run_cell "$SEED" epoch
    run_cell "$SEED" step
done

echo ""
echo "============================================================"
echo " 3d step-warmup vs 3c epoch-warmup comparison"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'

def agg(pattern):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, pattern.format(s=s), 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        return m, std, len(ppls), ppls
    return None

step = agg('phase3d_pa0.6_beta2.0_se0.5_stepwarmup_s{s}')
epoch = agg('phase3c_pa0.6_beta2.0_se0.5_s{s}')

print(f"{'config':>30}  {'mean':>8}  {'std':>6}  seeds")
if epoch:
    m, s, n, p = epoch
    print(f"{'3c  epoch warmup (baseline)':>30}  {m:>8.4f}  {s:>6.3f}  {p}")
if step:
    m, s, n, p = step
    print(f"{'3d  step warmup':>30}  {m:>8.4f}  {s:>6.3f}  {p}")

if step and epoch:
    delta = step[0] - epoch[0]
    sign = '+' if delta >= 0 else ''
    print(f"\nΔ (step − epoch): {sign}{delta:.4f} PPL")
    if abs(delta) < 0.15:
        print("  → within noise; per-epoch warmup is fine for NLP, no need to retrace ablation")
    elif delta < 0:
        print("  → step-based meaningfully BETTER; retest 500M ours with warmup_unit='step'")
    else:
        print("  → step-based WORSE; keep per-epoch warmup for final paper config")
EOF
echo "Done: $(date)"
