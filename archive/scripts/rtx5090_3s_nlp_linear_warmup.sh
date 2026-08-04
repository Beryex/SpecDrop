#!/bin/bash
###############################################################################
# RTX 5090 Script 3s: NLP mini-ablation — Phase S (linear warmup schedule
# test at Phase F root: uniform + scalar + wr=1.0 + no SE, pa=0.7).
#
# Motivation. We use cosine warmup as inherited from CIFAR; standard LR-warmup
# convention in NLP literature (Transformer / BERT / GPT / LLaMA) is linear.
# Phase A (wr=0) and Phase F (wr=1.0 cosine) sandwich showed wr SHAPE doesn't
# flip BEST_PA on NLP, but we have not tested the intermediate "linear"
# interpolation form. pa=0.7 is chosen because it's where cosine vs linear
# should differ most (middle of pa sweep, where wr's role in mitigating
# over-specialization matters).
#
# Search: pa=0.7 with linear warmup (3 seeds)
# Fixed:  uniform K=7 ffn=220, no SE, wr=1.0, 100M tokens, 10 epochs
# Compare to Phase F pa=0.7 (cosine) = 57.45 ± 0.13
#
# 3 runs, 1 per GPU seed, ~1.2h wall-clock per seed (parallel = 1.2h total).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseS_pa0.7_wr1.0_linear_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase S — linear warmup @ Phase F root"
echo " (uniform + scalar + wr=1.0 + no SE, pa=0.7, LINEAR schedule)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local SEED=$1
    local PA=0.7 PI=0.3
    local ENAME="phaseS_pa${PA}_wr1.0_linear_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI $PYTHON - <<'PYEOF'
import os, yaml

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': 220, 'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
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
print(f'[cfg-gen] Phase S: uniform ffn=220, no SE, wr=1.0 LINEAR, pa={PA} pi={PI}')
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    run_cell "$SEED"
done

echo ""
echo "============================================================"
echo " Phase S summary (linear warmup at pa=0.7)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
ppls = []
for s in (42, 123, 456):
    p = os.path.join(base, f'phaseS_pa0.7_wr1.0_linear_s{s}', 'results.json')
    if os.path.exists(p):
        ppls.append(json.load(open(p))['best_val_ppl'])
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((v-m)**2 for v in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
    print(f"Phase S (linear warmup, pa=0.7): {m:.2f} ± {std:.2f}  (n={len(ppls)})")
    print()
    print("Reference:")
    print(f"  Phase F pa=0.7 cosine warmup: 57.45 ± 0.13")
    delta = m - 57.45
    print(f"  Δ (linear − cosine): {delta:+.2f} PPL")
    if delta < -0.15:
        print("  → Linear warmup meaningfully helps at pa=0.7.")
    elif delta > 0.15:
        print("  → Cosine warmup is better at pa=0.7.")
    else:
        print("  → Schedule shape doesn't matter (within noise).")
EOF
echo "Done: $(date)"
