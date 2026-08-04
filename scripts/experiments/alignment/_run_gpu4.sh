#!/bin/bash
# GPU 4 — rtx5090_15 alignment dispatch (~230 min total).
#
# Cells (13):
#   L13-L14  LoRA   LoRAMoE s456 + MoCLE s42                                   (~50 min × 2)
#   V10-V11  ViT    soft_moe_vit s123/s456                                       (~30 min × 2)
#   N10,N14-N19  NLP    no_routing_se05 s456 + Hash×3 + Demix×3                 (~10 min × 7)
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L13,L14,V10,V11,N10,N14,N15,N16,N17,N18,N19,N23,L17"
exec bash scripts/experiments/alignment/run.sh
