#!/bin/bash
###############################################################################
# RTX 5090 Script 2/4 (FAITHFUL): CIFAR-100 × ResNet-110, native baselines only.
#
# After pivot to 3-setting native comparison (2026-04-12+), CIFAR-100 × ResNet-110
# compares ONLY baselines whose original paper ran on CIFAR ResNet:
#
#  1. ResNet-110 (dense, reference)
#  2. Stochastic Depth (Huang ECCV 2016, dense ResNet-110 + block-level drop) — NATIVE
#  3. Example-Tied Dropout (Maini ICML 2023, dense ResNet-110 + per-example mask) — NEW
#  4. Contextual Dropout (Fan ICLR 2021, Gaussian variant on ResNet-110) — NEW
#  5. MultiBranchResNet-110 no-routing (architectural reference)
#  6. Soft SpecDrop (OURS) — pa=0.7, wr=1.0, no SE, bc=[4,7,14]
#
# Setting (identical for ALL):
#   Optimizer: SGD lr=0.1, mom=0.9, wd=5e-4
#   Schedule:  Linear warmup (5 ep) + cosine annealing (195 ep)
#   Epochs:    200, Batch: 128, Seeds: 42 / 123 / 456
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_cifar100_faithful"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " CIFAR-100 × ResNet-110 FAITHFUL (7 methods × ${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 200, 'batch_size': 128, 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_schedule': 'cosine', 'warmup_epochs': 5"
COMMON_DATA="'dataset': 'cifar100', 'data_dir': './data_cache', 'num_workers': 4"
FB_MODEL="'type': 'multi_branch_resnet110', 'base_channels': 16, 'num_branches': 20, 'branch_channels': [4, 7, 14], 'num_blocks': 18, 'num_classes': 100"

run_experiment() {
    local NAME=$1 CONFIG_GEN=$2
    for SEED in "${SEEDS[@]}"; do
        local ENAME="${NAME}_s${SEED}"
        local ODIR="${OUTDIR_BASE}/${ENAME}"
        if [ -f "$ODIR/results.json" ]; then
            echo "  $ENAME — DONE, skipping"; continue
        fi
        mkdir -p "$ODIR"
        echo "  Running $ENAME ... ($(date))"
        $PYTHON -c "
import yaml
$CONFIG_GEN
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
yaml.dump(cfg, open('$OUTDIR_BASE/_tmp_s${SEED}.yaml', 'w'))
"
        $PYTHON run.py --wandb --config "$OUTDIR_BASE/_tmp_s${SEED}.yaml" --device $DEVICE 2>&1 | tee "${OUTDIR_BASE}/${ENAME}.log"
        echo "  $ENAME finished: $(date)"
    done
}

echo "[1/7] ResNet-110 (dense, reference)"
run_experiment "resnet110" "
cfg = {
    'model': {'type': 'resnet', 'base_channels': 16, 'num_blocks': 18, 'num_classes': 100, 'dropout_rate': 0.0},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[2/7] Stochastic Depth (Huang 2016, DENSE ResNet-110, sd_rate=0.5)"
run_experiment "stoch_depth" "
cfg = {
    'model': {'type': 'resnet', 'base_channels': 16, 'num_blocks': 18, 'num_classes': 100, 'dropout_rate': 0.0, 'stochastic_depth_rate': 0.5},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[3/7] Example-Tied Dropout (Maini 2023, α_mem=0.5, keep=0.5)"
run_experiment "et_dropout" "
cfg = {
    'model': {'type': 'example_tied_dropout_resnet110', 'num_classes': 100, 'num_examples': 50000, 'alpha_mem': 0.5, 'mem_keep_rate': 0.5},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[4/7] Contextual Dropout (Fan 2021, Gaussian, γ=8)"
run_experiment "ctx_dropout" "
cfg = {
    'model': {'type': 'contextual_dropout_resnet110', 'num_classes': 100, 'reduction': 8},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[5/7] MultiBranchResNet-110 no-routing (architectural ref)"
run_experiment "no_routing" "
cfg = {
    'model': {$FB_MODEL},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[6/7] Soft SpecDrop (OURS) — pa=0.7, wr=1.0, no SE"
run_experiment "ours" "
cfg = {
    'model': {$FB_MODEL},
    'algorithm': {'type': 'soft_specdrop', 'p_active': 0.7, 'p_inactive': 0.3, 'assignment': 'round_robin', 'warmup_ratio': 1.0},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[7/7] HardCategory (DEMix-style one-hot routing, metadata-aware baseline)"
for SEED in "${SEEDS[@]}"; do
    ODIR="./outputs/cv_hard_category_k20/s${SEED}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  hard_category_s${SEED} — DONE, skipping"; continue
    fi
    mkdir -p "$ODIR"
    $PYTHON -c "
import yaml
cfg = yaml.safe_load(open('configs/cv/hard_category_k20.yaml'))
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = 'hard_category_s${SEED}'
yaml.dump(cfg, open('${OUTDIR_BASE}/_tmp_hc_s${SEED}.yaml', 'w'))
"
    $PYTHON run.py --wandb --config "${OUTDIR_BASE}/_tmp_hc_s${SEED}.yaml" --device $DEVICE 2>&1 | tee "./outputs/cv_hard_category_k20/hard_category_s${SEED}.log"
done

echo ""
echo "============================================================"
echo " CIFAR-100 FAITHFUL Results (ResNet-110 scale, ~1.7M params)"
echo "============================================================"
$PYTHON -c "
import json, os
methods = ['resnet110','stoch_depth','et_dropout','ctx_dropout','no_routing','hard_category','ours']
labels  = ['ResNet-110','Stoch Depth','ETD','Ctx Dropout','No Routing','HardCategory','Ours']
for m, l in zip(methods, labels):
    accs = []
    for s in [42, 123, 456]:
        path = (f'./outputs/cv_hard_category_k20/s{s}/results.json'
                if m == 'hard_category' else f'$OUTDIR_BASE/{m}_s{s}/results.json')
        if os.path.exists(path):
            accs.append(json.load(open(path))['best_top1'])
    if accs:
        mean = sum(accs)/len(accs)
        std = (sum((x-mean)**2 for x in accs)/len(accs))**0.5
        print(f'  {l:15s}: {mean:.2f} +/- {std:.2f}  (n={len(accs)})')
"
echo "Done: $(date)"
