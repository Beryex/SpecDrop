#!/bin/bash
# GPU 5 — seed 456. See _run_gpu_ViT_0.sh header for the full chain description,
# knobs, and timing notes.
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
set -e
mkdir -p outputs

SEED=456

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_pa.sh 2>&1 | tee outputs/log_gpu_ViT_2_5a_s${SEED}.txt

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_beta.sh 2>&1 | tee outputs/log_gpu_ViT_2_5b_s${SEED}.txt

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_se.sh 2>&1 | tee outputs/log_gpu_ViT_2_5c_s${SEED}.txt

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/main_table.sh 2>&1 | tee outputs/log_gpu_ViT_2_6_s${SEED}.txt

echo "GPU 5 (seed $SEED) ViT 5a + 5b + 5c + 6 done: $(date)"
