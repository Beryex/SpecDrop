#!/bin/bash
###############################################################################
# RTX 5090 Script 3a: NLP mini-ablation — Phase A (p_active sweep).
#
# Narrowed CIFAR-inspired ablation on SlimPajama at REDUCED scale (100M tokens)
# to find the NLP-specific optimum for (pa, wr, SE_ratio) under a tight budget.
# CIFAR Phase A found pa=0.7 optimal at K=20; we re-sweep at NLP's K=7 since
# each branch's post-merge weight is 2.5× larger, so the inverted-U may shift.
#
# Phase A search: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa
# Fixed:        K=7, ffn=220 (no SE), wr=0.0, 100M tokens, 10 epochs, 3 seeds
#
# CIFAR's promising region was narrowed from 7 values to 6: pa=0.95 dropped
# (monotone tail in CIFAR); pa=0.5 KEPT because CIFAR showed a cliff there
# (62.00 Top-1) and NLP's softer-specialization regime might shift the
# cliff location.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseA_pa{pa}_pi{pi}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase A — pa sweep (${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

# 100M tokens, otherwise identical to rtx5090_4_nlp_faithful setting.
COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"
BASE_MODEL="'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseA_pa${PA}_pi${PI}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    local TMPYAML="${ODIR}/_tmp.yaml"
    $PYTHON -c "
import yaml
cfg = {
    'model': {$BASE_MODEL},
    'algorithm': {'type': 'soft_specdrop', 'p_active': $PA, 'p_inactive': $PI, 'assignment': 'round_robin', 'warmup_ratio': 0.0},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
with open('$TMPYAML', 'w') as f:
    yaml.dump(cfg, f)
"
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "$TMPYAML" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase A] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase A summary (once all 3 seeds complete)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'pa':>5} {'pi':>5}  {'mean PPL':>10}  {'std':>6}  n seeds")
rows = []
for pa in ['0.5','0.6','0.7','0.8','0.9','1.0']:
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseA_pa{pa}_pi{pi}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls) / len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
if rows:
    complete = [r for r in rows if r[4] == 3]
    if complete:
        best = min(complete, key=lambda r: r[2])
        print(f"\nBEST_PA (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
EOF
echo "Done: $(date)"
