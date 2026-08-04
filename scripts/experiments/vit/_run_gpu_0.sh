#!/bin/bash
# GPU 0 — seed 42. ViT ImageNet ablation chain + main table:
#   (1) rtx5090_5a: pa sweep @ SE=1.0, β=1 (6 pa × 1 seed = 6 runs)
#   (2) rtx5090_5b: β sweep @ (best_pa, SE=1.0) (3 new β × 1 seed = 3 runs;
#       barriers across GPUs on 5a, writes _best_pa.txt)
#   (3) rtx5090_5c: SE sweep @ (best_pa, best_β) (3 new SE × 1 seed = 3 runs;
#       barriers on 5b, writes _best_beta.txt + _best_se.txt)
#   (4) rtx5090_6: faithful main table (7 methods × 1 seed, reads _best_*.txt
#       markers so ours locks in the argmax config automatically)
#
# NLP SlimPajama cells are frozen (no re-runs);
# this GPU script drives the ViT track. Post-train multi-branch LoRA is a
# parallel P0 track starting separately once this kicks off.
#
# All ViT runs use 100 epochs + 5ep linear warmup (DeiT recipe, compute-
# budget version — DeiT paper gold is 300ep but not shipped here).
#
# Timing (≈ at 100ep on 5090, per GPU):
#   5a:  6 runs × ~10h  = ~60h
#   5b:  3 runs × ~10h  = ~30h
#   5c:  3 runs × ~10h  = ~30h
#   6:   7 runs × ~10h  = ~70h
#   Sequential wall: ~8 days per GPU. 3 GPUs in parallel: ~8 days (each GPU
#   runs its own seed through the same chain; barriers synchronize at 5b/5c).
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
set -e
mkdir -p outputs

SEED=42

# ── ViT mini-ablation 5a → 5b → 5c ─────────────────────────────────────────
SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_pa.sh 2>&1 | tee outputs/log_gpu_ViT_0_5a_s${SEED}.txt

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_beta.sh 2>&1 | tee outputs/log_gpu_ViT_0_5b_s${SEED}.txt

SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/ablation_se.sh 2>&1 | tee outputs/log_gpu_ViT_0_5c_s${SEED}.txt

# ── ViT faithful main table (auto-reads _best_*.txt markers from 5c) ───────
SEEDS_OVERRIDE=$SEED \
    bash scripts/experiments/vit/main_table.sh 2>&1 | tee outputs/log_gpu_ViT_0_6_s${SEED}.txt

echo "GPU 3 (seed $SEED) ViT 5a + 5b + 5c + 6 done: $(date)"
