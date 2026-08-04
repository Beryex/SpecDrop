#!/bin/bash
###############################################################################
# Ablation Phase A: p_active sweep
#
# Sweep p_active/p_inactive: (1.0,0.0) (0.95,0.05) (0.9,0.1) (0.8,0.2) (0.7,0.3) (0.6,0.4) (0.5,0.5)
# Fixed: shared_expert=True, warmup_ratio=0
#
# Setting:
#   Model:     MultiBranchResNet110 K=20, branch_channels=[3,7,14] + 1.0x shared expert, ~1.72M params
#   Dataset:   CIFAR-100 (20 superclasses, 50K train / 10K test)
#   Optimizer: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
#   Schedule:  Linear LR warmup (5 epochs) + cosine annealing (195 epochs)
#   Epochs:    200
#   Batch:     128
#
# Usage (3 GPUs in parallel):
#   CUDA_VISIBLE_DEVICES=0 SEEDS_OVERRIDE=42  bash scripts/experiments/cifar/ablation_pa.sh 2>&1 | tee log_1a_s42.txt
#   CUDA_VISIBLE_DEVICES=1 SEEDS_OVERRIDE=123 bash scripts/experiments/cifar/ablation_pa.sh 2>&1 | tee log_1a_s123.txt
#   CUDA_VISIBLE_DEVICES=2 SEEDS_OVERRIDE=456 bash scripts/experiments/cifar/ablation_pa.sh 2>&1 | tee log_1a_s456.txt
#
# After all 3 finish, check results:
#   python -c "
#   import json, os
#   for pa in ['0.95','0.9','0.8','0.7','0.6','0.5']:
#       accs = []
#       for s in [42,123,456]:
#           pi = str(round(1-float(pa),2))
#           p = f'outputs/rtx5090_ablation/ablation_pa{pa}_pi{pi}_s{s}/results.json'
#           if os.path.exists(p): accs.append(json.load(open(p))['best_top1'])
#       if accs:
#           m=sum(accs)/len(accs); std=(sum((x-m)**2 for x in accs)/len(accs))**0.5
#           print(f'  pa={pa}: {m:.2f} +/- {std:.2f} ({len(accs)} seeds)')
#   "
#
# Then use the best p_active for Phase B.
#
# Output: outputs/rtx5090_ablation/ablation_pa{pa}_pi{pi}_s{seed}/results.json
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

echo "============================================================"
echo " Ablation Phase A: p_active sweep (seeds: ${SEEDS[*]})"
echo " $(date)"
echo "============================================================"

run_one() {
    local NAME=$1 PA=$2 PI=$3 SEED=$4
    local ENAME="${NAME}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"
        return
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
        'branch_channels': [3, 7, 14],
        'num_blocks': 18,
        'num_classes': 100,
        'shared_expert': True,
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': $PA,
        'p_inactive': $PI,
        'assignment': 'round_robin',
        'warmup_ratio': 0,
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
}

for PAIR in "1.0,0.0" "0.95,0.05" "0.9,0.1" "0.8,0.2" "0.7,0.3" "0.6,0.4" "0.5,0.5"; do
    IFS=',' read -r PA PI <<< "$PAIR"
    NAME="ablation_pa${PA}_pi${PI}"
    echo "[$PA / $PI]"
    for SEED in "${SEEDS[@]}"; do
        run_one "$NAME" "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "=== Phase A Results ==="
$PYTHON -c "
import json, os
for pa in ['1.0', '0.95', '0.9', '0.8', '0.7', '0.6', '0.5']:
    accs = []
    for s in [42, 123, 456]:
        pi = str(round(1.0 - float(pa), 2))
        path = f'$OUTDIR_BASE/ablation_pa{pa}_pi{pi}_s{s}/results.json'
        if os.path.exists(path):
            accs.append(json.load(open(path))['best_top1'])
    if accs:
        mean = sum(accs)/len(accs)
        std = (sum((x-mean)**2 for x in accs)/len(accs))**0.5
        print(f'  pa={pa}: {mean:.2f} +/- {std:.2f} ({len(accs)} seeds)')
"
echo "Done: $(date)"
