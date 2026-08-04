#!/bin/bash
# GPU 3 — rtx5090_15 alignment dispatch (~242 min total).
#
# Cells (10):
#   L10-L12  LoRA   HydraLoRA s456 + LoRAMoE s42/s123                          (~50 min × 3)
#   V08-V09  ViT    mbvit_no_routing_se s456 + soft_moe s42                    (~30 min × 2)
#   N07-N09  NLP    no_routing_se s123/s456 + no_routing_se05 s123             (~10 min × 3)
#   C03      CIFAR  hard_category s456                                          (~ 2 min × 1)
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L10,L11,L12,V08,V09,N07,N08,N09,C03,C09"
exec bash scripts/experiments/alignment/run.sh
