#!/bin/bash
# Batch-runs diagnose_nlp_specialization.py on the 5 most interesting checkpoints
# from the Phase W campaign. Each diagnostic: baseline eval + K branch-ablation
# passes = K+1 forward passes × --max_batches val batches. CPU-feasible; on GPU
# ~2-5 min each.
#
# Runs produce outputs/analysis/nlp_diag/<run_name>.json + printed summary.
# Rsync those JSONs back locally for aggregation / interpretation.
#
# Usage:  bash scripts/run_nlp_diag_batch.sh [max_batches=50]

set -e
MAX_BATCHES=${1:-50}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
BASE="outputs/rtx5090_nlp_mini_ablation"
SEED=${SEED:-42}

# 5 checkpoints forming a clean comparison:
#   (1) Phase D  pa=0.6 — scalar routing on domains; the absolute PPL floor.
#                          Specialization with no per-cat mechanism.
#   (2) Phase P  pa=0.6 — per-cat routing on domain labels. Strong baseline.
#   (3) Phase W  pa=0.6 — per-cat routing on semantic cluster (k=7) labels.
#                          Does it specialize on clusters?
#   (4) Phase W  pa=0.7 — same as (3) but higher pa. Does stricter routing
#                          amplify cluster boundary noise (our hypothesis)?
#   (5) Phase Z  k=3 pa=0.6 — coarse 3-cluster. Does specialization concentrate?
RUNS=(
    "phaseD_pa0.6_pi0.4_s${SEED}"
    "phaseP_pa0.6_wr1.0_s${SEED}"
    "phaseW_cluster_pa0.6_wr1.0_s${SEED}"
    "phaseW_cluster_pa0.7_wr1.0_s${SEED}"
    "phaseZ_cluster_k3_pa0.6_wr1.0_s${SEED}"
)

mkdir -p outputs/analysis/nlp_diag

for r in "${RUNS[@]}"; do
    run_dir="${BASE}/${r}"
    if [ ! -d "$run_dir" ] || [ ! -f "${run_dir}/best.pt" ]; then
        echo "skip ${r}: missing run dir or best.pt"
        continue
    fi
    out="outputs/analysis/nlp_diag/${r}.json"
    if [ -f "$out" ]; then
        echo "skip ${r}: ${out} already exists"
        continue
    fi
    echo ""
    echo "============================================================"
    echo " Diagnostic: ${r}"
    echo "============================================================"
    $PYTHON scripts/diagnose_nlp_specialization.py \
        --run_dir "$run_dir" \
        --max_batches "$MAX_BATCHES" \
        --device "$DEVICE" \
        --out_json "$out" \
        2>&1 | tee "outputs/analysis/nlp_diag/${r}.log"
done

echo ""
echo "============================================================"
echo " Batch done. JSONs at outputs/analysis/nlp_diag/"
echo " To aggregate locally: python scripts/aggregate_alignment.py"
echo "============================================================"
