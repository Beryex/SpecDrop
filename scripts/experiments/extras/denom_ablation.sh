#!/bin/bash
###############################################################################
# Mask x denominator ablation (paper App. A.3, Tab. 6): four corners
# {stochastic, random-dropout} x {fixed S, stochastic sum_k m_k} at the
# ResNet-110 / CIFAR-100 scale, 3 seeds each. Output dirs match
# scripts/summarize_e3_denom.py's expectations. Skips completed cells.
#
# Usage:  bash scripts/experiments/extras/denom_ablation.sh
#         SEEDS_OVERRIDE=42 CUDA_VISIBLE_DEVICES=0 bash ... (per-seed parallel)
# Summary afterwards:  python scripts/summarize_e3_denom.py
###############################################################################
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p outputs/rtx5090_denom_ablation

for CFG in stoch_fixed stoch_naive rand_fixed rand_naive; do
    for SEED in "${SEEDS[@]}"; do
        ODIR="outputs/rtx5090_denom_ablation/${CFG}_s${SEED}"
        if [ -f "$ODIR/results.json" ]; then
            echo "  [${CFG} s${SEED}] DONE, skipping"
            continue
        fi
        echo "  [${CFG} s${SEED}] running ... ($(date))"
        mkdir -p "$ODIR"
        $PYTHON run.py --config "configs/cv/${CFG}.yaml" \
            --output_dir "$ODIR" --seed "$SEED" --device "$DEVICE" --no-wandb \
            2>&1 | tee "$ODIR/train.log" || \
            echo "  [${CFG} s${SEED}] FAILED"
        echo "  [${CFG} s${SEED}] finished: $(date)"
    done
done
echo "All cells attempted. Summarize: $PYTHON scripts/summarize_e3_denom.py"
