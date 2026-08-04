#!/bin/bash
# GPU 0 — rtx5090_15 alignment dispatch (~240 min total).
#
# Cells (10):
#   L01-L03  LoRA   ours / mb_lora_no_routing × s123/s456                     (~50 min × 3)
#   V01-V03  ViT    ours_vit s123/s456 + mbvit_no_routing s42                  (~30 min × 3)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L01,L02,L03,V01,V02,V03,C04,C05,C06,V15"
exec bash scripts/experiments/alignment/run.sh
