#!/bin/bash
###############################################################################
# Ablation Phase C: shared expert dimension ratio sweep
#
# Sweep SE ratio: 0.25x, 0.5x, 1.0x, 2.0x, 4.0x of branch channels
# All param-matched to ResNet-110 (~1.74M)
# Fixed: p_active=BEST_PA, warmup_ratio=BEST_WR (from Phase A & B)
#
# REQUIRES: BEST_PA and BEST_WR env vars
#
# Usage (3 GPUs in parallel):
#   CUDA_VISIBLE_DEVICES=0 BEST_PA=0.9 BEST_WR=0 SEEDS_OVERRIDE=42  bash scripts/experiments/cifar/ablation_se.sh 2>&1 | tee log_1c_s42.txt
#   CUDA_VISIBLE_DEVICES=1 BEST_PA=0.9 BEST_WR=0 SEEDS_OVERRIDE=123 bash scripts/experiments/cifar/ablation_se.sh 2>&1 | tee log_1c_s123.txt
#   CUDA_VISIBLE_DEVICES=2 BEST_PA=0.9 BEST_WR=0 SEEDS_OVERRIDE=456 bash scripts/experiments/cifar/ablation_se.sh 2>&1 | tee log_1c_s456.txt
#
# SE ratio configs (all ~1.74M params):
#   SE 0.25x:  3 SE blocks, bc=[4,7,14] → 1.735M (0.999x)
#   SE 0.5x:   7 SE blocks, bc=[4,7,14] → 1.754M (1.010x)
#   SE 1.0x:  18 SE blocks, bc=[3,7,14] → 1.755M (1.010x)  [default]
#   SE 2.0x:  35 SE blocks, bc=[3,6,14] → 1.739M (1.001x)
#   SE 4.0x:  75 SE blocks, bc=[3,6,13] → 1.706M (0.983x)
#
# Output: outputs/rtx5090_ablation/ablation_se_ratio_{ratio}x_pa{BEST_PA}_wr{BEST_WR}_s{seed}/results.json
###############################################################################

# Validate env vars
if [ -z "$BEST_PA" ]; then
    echo "ERROR: BEST_PA env var required. Set it to best p_active from Phase A."
    echo "Example: BEST_PA=0.9 BEST_WR=0 bash $0"
    exit 1
fi
if [ -z "$BEST_WR" ]; then
    echo "ERROR: BEST_WR env var required. Set it to best warmup_ratio from Phase B."
    echo "Example: BEST_PA=0.9 BEST_WR=0 bash $0"
    exit 1
fi
BEST_PI=$(python -c "print(round(1.0 - float('$BEST_PA'), 2))")

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

echo "============================================================"
echo " Ablation Phase C: SE ratio sweep (pa=$BEST_PA, wr=$BEST_WR, seeds: ${SEEDS[*]})"
echo " $(date)"
echo "============================================================"

# SE ratio configs (all param-matched to ResNet-110 ~1.74M): vary shared_expert_blocks, adjust branch_channels for param matching
declare -A SE_BLOCKS SE_BC_CFG
SE_BLOCKS["0.25"]=3    ; SE_BC_CFG["0.25"]="4, 7, 14"
SE_BLOCKS["0.5"]=7    ; SE_BC_CFG["0.5"]="4, 7, 14"
SE_BLOCKS["1.0"]=18   ; SE_BC_CFG["1.0"]="3, 7, 14"
SE_BLOCKS["2.0"]=35   ; SE_BC_CFG["2.0"]="3, 6, 14"
SE_BLOCKS["4.0"]=75   ; SE_BC_CFG["4.0"]="3, 6, 13"

# First: SE=0 baseline (no shared expert)
echo "[SE=0 (no shared expert), bc=[4,7,14], pa=${BEST_PA}, wr=${BEST_WR}]"
for SEED in "${SEEDS[@]}"; do
    ENAME="ablation_se_ratio_0x_pa${BEST_PA}_wr${BEST_WR}_s${SEED}"
    ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"
        continue
    fi
    mkdir -p "$ODIR"
    echo "  Running $ENAME ... ($(date))"
    $PYTHON -c "
import yaml
cfg = {
    'experiment_name': '$ENAME',
    'model': {
        'type': 'multi_branch_resnet110',
        'base_channels': 16,
        'num_branches': 20,
        'branch_channels': [4, 7, 14],
        'num_blocks': 18,
        'num_classes': 100,
        'shared_expert': False,
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': $BEST_PA,
        'p_inactive': $BEST_PI,
        'assignment': 'round_robin',
        'warmup_ratio': $BEST_WR,
    },
    'training': {
        'epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_schedule': 'cosine',
        'warmup_epochs': 5,
    },
    'data': {
        'dataset': 'cifar100',
        'data_dir': './data_cache',
        'num_workers': 4,
    },
    'output_dir': '$ODIR',
    'seed': $SEED,
}
yaml.dump(cfg, open('$OUTDIR_BASE/_tmp_s${SEED}.yaml', 'w'))
"
    $PYTHON run.py --wandb --config "$OUTDIR_BASE/_tmp_s${SEED}.yaml" --device $DEVICE \
        2>&1 | tee "${OUTDIR_BASE}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
done

# Then: SE ratio sweep (with shared expert)
for SE_RATIO in 0.25 0.5 1.0 2.0 4.0; do
    SE_BLK=${SE_BLOCKS[$SE_RATIO]}
    BC_STR=${SE_BC_CFG[$SE_RATIO]}
    NAME="ablation_se_ratio_${SE_RATIO}x_pa${BEST_PA}_wr${BEST_WR}"
    echo "[SE ratio=${SE_RATIO}x, se_blocks=${SE_BLK}, bc=[${BC_STR}], pa=${BEST_PA}, wr=${BEST_WR}]"

    for SEED in "${SEEDS[@]}"; do
        ENAME="${NAME}_s${SEED}"
        ODIR="${OUTDIR_BASE}/${ENAME}"
        if [ -f "$ODIR/results.json" ]; then
            echo "  $ENAME — DONE, skipping"
            continue
        fi
        mkdir -p "$ODIR"
        echo "  Running $ENAME ... ($(date))"
        $PYTHON -c "
import yaml
cfg = {
    'experiment_name': '$ENAME',
    'model': {
        'type': 'multi_branch_resnet110',
        'base_channels': 16,
        'num_branches': 20,
        'branch_channels': [${BC_STR}],
        'shared_expert_blocks': ${SE_BLK},
        'num_blocks': 18,
        'num_classes': 100,
        'shared_expert': True,
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': $BEST_PA,
        'p_inactive': $BEST_PI,
        'assignment': 'round_robin',
        'warmup_ratio': $BEST_WR,
    },
    'training': {
        'epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'lr_schedule': 'cosine',
        'warmup_epochs': 5,
    },
    'data': {
        'dataset': 'cifar100',
        'data_dir': './data_cache',
        'num_workers': 4,
    },
    'output_dir': '$ODIR',
    'seed': $SEED,
}
yaml.dump(cfg, open('$OUTDIR_BASE/_tmp_s${SEED}.yaml', 'w'))
"
        $PYTHON run.py --wandb --config "$OUTDIR_BASE/_tmp_s${SEED}.yaml" --device $DEVICE \
            2>&1 | tee "${OUTDIR_BASE}/${ENAME}.log"
        echo "  $ENAME finished: $(date)"
    done
done

echo ""
echo "=== Phase C Results ==="
$PYTHON -c "
import json, os
for r in ['0', '0.25', '0.5', '1.0', '2.0', '4.0']:
    accs = []
    for s in [42, 123, 456]:
        path = f'$OUTDIR_BASE/ablation_se_ratio_{r}x_pa${BEST_PA}_wr${BEST_WR}_s{s}/results.json'
        if os.path.exists(path):
            accs.append(json.load(open(path))['best_top1'])
    if accs:
        mean = sum(accs)/len(accs)
        std = (sum((x-mean)**2 for x in accs)/len(accs))**0.5
        print(f'  SE ratio={r}x: {mean:.2f} +/- {std:.2f} ({len(accs)} seeds)')
"
echo "Done: $(date)"
