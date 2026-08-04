#!/bin/bash
# GPU 1 — rtx5090_15 alignment dispatch (~242 min total).
#
# Cells (10):
#   L04-L06  LoRA   mb_lora_no_routing s456 + se1.0 s42/s123                  (~50 min × 3)
#   V04-V05  ViT    mbvit_no_routing s123/s456                                  (~30 min × 2)
#   N01-N03  NLP    ours_phaseP s123/s456 + no_routing s42                     (~10 min × 3)
#   C01      CIFAR  hard_category s42                                           (~ 2 min × 1)
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L04,L05,L06,V04,V05,N01,N02,N03,C01,C07"
exec bash scripts/experiments/alignment/run.sh
