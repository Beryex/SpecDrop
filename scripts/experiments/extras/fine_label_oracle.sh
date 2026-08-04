#!/bin/bash
###############################################################################
# CIFAR fine-label oracle (paper App. E.2): Soft SpecDrop with M=100 fine
# labels routed over K=20 branches (5 labels/branch, deliberately leaky).
# 3 seeds; ~2h/seed on an RTX 5090. Skips completed cells.
#
# Usage:  bash scripts/experiments/extras/fine_label_oracle.sh
#         SEEDS_OVERRIDE=42 CUDA_VISIBLE_DEVICES=0 bash ... (per-seed parallel)
###############################################################################
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p outputs/cv_ours_fine

for SEED in "${SEEDS[@]}"; do
    ODIR="outputs/cv_ours_fine/s${SEED}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  [ours_fine s${SEED}] DONE, skipping"
        continue
    fi
    echo "  [ours_fine s${SEED}] running ... ($(date))"
    mkdir -p "$ODIR"
    $PYTHON run.py --config configs/cv/ours_fine.yaml \
        --output_dir "$ODIR" --seed "$SEED" --device "$DEVICE" --no-wandb \
        2>&1 | tee "$ODIR/train.log" || \
        echo "  [ours_fine s${SEED}] FAILED"
    echo "  [ours_fine s${SEED}] finished: $(date)"
done
