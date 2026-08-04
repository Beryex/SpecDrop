#!/bin/bash
# GPU 5 — rtx5090_15 alignment dispatch (~250 min total).
#
# Cells (13):
#   L15-L16  LoRA   MoCLE s123/s456                                              (~50 min × 2)
#   V12-V14  ViT    mod_squad_vit s42/s123/s456                                 (~30 min × 3)
#   N11-N13,N20-N22  NLP    Switch×3 + SMoE-Dropout×3                          (~10 min × 6)
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L15,L16,V12,V13,V14,N11,N12,N13,N20,N21,N22,N24,L18"
exec bash scripts/experiments/alignment/run.sh
