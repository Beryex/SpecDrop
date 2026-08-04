#!/bin/bash
# Helper: 3-tier skip / auto-recover logic for training cells.
#
# Usage (after sourcing this file):
#     ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON"
#     rc=$?
#     [ $rc -eq 0 ] && { return; }          # inside a function
#     [ $rc -eq 0 ] && { continue; }         # inside a for loop
#     # rc=1 means the cell needs full training; caller proceeds
#
# Cell state machine (keyed off `results.json`, which trainer writes ONLY
# after all epochs finish — see trainer_lora.py::finalize_results):
#   (a) results.json exists AND eval_rouge_l NOT None → fully DONE, skip.
#   (b) results.json exists AND eval_rouge_l is None → training completed,
#       only ROUGE is missing/failed; rerun ROUGE inline, then skip.
#   (c) results.json missing → training NOT finished (never started OR
#       crashed mid-way). Must retrain from scratch, regardless of whether
#       a stale best.pt exists (best.pt is written every epoch the eval
#       loss improves — it's NOT a completion signal).
#
# If (b) fails, we still skip — training already burned, user can retry
# rerun_rouge manually. Re-training would waste hours on a finished model.
#
# Dependencies: python interpreter that can load `json` and run
# `scripts/rerun_rouge.py` (i.e., env with torch/transformers).

ensure_cell_or_rerun_rouge() {
    local enam="$1"
    local odir="$2"
    local pyth="${3:-python}"

    # No results.json → training didn't finish. Fall through to tier (c).
    # (Guard against stale best.pt from a crashed run — DON'T rerun ROUGE
    # on a partially-trained model.)
    if [ ! -f "$odir/results.json" ]; then
        return 1
    fi

    # Tier (a): training finished AND ROUGE populated
    local rouge_ok
    rouge_ok=$("$pyth" -c "
import json
try:
    with open('$odir/results.json') as f: r = json.load(f)
    print('yes' if r.get('eval_rouge_l') is not None else 'no')
except Exception:
    print('no')
" 2>/dev/null)
    if [ "$rouge_ok" = "yes" ]; then
        echo "  $enam — DONE (training + ROUGE), skipping"
        return 0
    fi

    # Tier (b): training finished (results.json present) but ROUGE is
    # None — e.g. eval crashed with the dtype bug or shape-mismatch bug.
    # best.pt is guaranteed to be the final-epoch checkpoint here. Safe
    # to rerun ROUGE.
    echo "  $enam — training finished but ROUGE missing; running rerun_rouge inline..."
    if "$pyth" scripts/rerun_rouge.py "$odir"; then
        echo "  $enam — ROUGE rerun OK, cell complete, skipping"
    else
        echo "  $enam — ROUGE rerun FAILED; NOT re-training (training already complete). "
        echo "                  Fix rerun_rouge.py + retry \`python scripts/rerun_rouge.py $odir\` manually."
    fi
    return 0  # either way, don't re-train
}
