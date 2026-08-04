#!/bin/bash
# GPU 2 — rtx5090_15 alignment dispatch (~242 min total).
#
# Cells (10):
#   L07-L09  LoRA   se1.0 s456 + HydraLoRA s42/s123                            (~50 min × 3)
#   V06-V07  ViT    mbvit_no_routing_se s42/s123                                 (~30 min × 2)
#   N04-N06  NLP    no_routing s123/s456 + no_routing_se s42                   (~10 min × 3)
#   C02      CIFAR  hard_category s123                                          (~ 2 min × 1)
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

PYTHON=${PYTHON:-python}
export DEVICE=cuda
export CELLS_FILTER="L07,L08,L09,V06,V07,N04,N05,N06,C02,C08"
exec bash scripts/experiments/alignment/run.sh
