#!/bin/bash
###############################################################################
# Category-free inference robustness + specialization diagnostics
#
# Two-part follow-up addressing the cluster_id-at-inference disclosure and strengthening the
# cross-setting specialization claim.
#
# === P0: Category-free inference robustness (4 settings × 3 seeds = 12 cells)
#
# For each (setting, seed) pair, load ours' best.pt, swap algorithm to
# NoDropout (uniform mask 1/K, S=K, output = ensemble mean = mech-OFF), run
# eval pipeline. Δ = uniform_metric − src_metric quantifies cost of needing
# cluster_id at inference (paper disclosure C2).
#
# Per-seed dispatch via SEEDS_OVERRIDE (mirrors all other rtx5090_X scripts):
#   GPU 0 (s42)  : CIFAR + ViT + NLP + LoRA           ≈ 6h wall
#   GPU 1 (s123) : CIFAR + ViT + NLP + LoRA           ≈ 6h wall
#   GPU 2 (s456) : CIFAR + ViT + NLP + LoRA           ≈ 6h wall
#
# === P1: Specialization diagnostics (single-seed, s42 only — heatmaps)
#
# Adapted from scripts/diagnose_nlp_specialization.py:
#   - ViT  : per-supercategory × per-branch top1 ablation matrix (46×46)
#   - LoRA : per-cluster × per-branch ROUGE-L ablation matrix (20×20)
# Single-seed because output is a heatmap (figure), not a statistical claim.
# Only runs on GPU 0 (SEEDS_OVERRIDE=42) — others skip via the seed gate.
#
# Outputs:
#   outputs/eval_uniform_mask/<setting>_s<seed>.json   (P0; 12 files total)
#   outputs/analysis/vit_diag/ours_vit_s42.json        (P1 ViT)
#   outputs/analysis/lora_diag/ours_s42.json           (P1 LoRA)
#
# Skip-logic: each cell checks its own output JSON; if present, skip.
#
# When chained from _run_gpu*.sh, all already-completed cells auto-skip.
###############################################################################

PYTHON=${PYTHON:-python}
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p outputs/eval_uniform_mask outputs/analysis/vit_diag outputs/analysis/lora_diag

echo "============================================================"
echo " RTX 5090 Script 10: P0 uniform-mask + P1 specialization diagnostics"
echo " Seeds: ${SEEDS[*]}  $(date)"
echo "============================================================"

# ─── P0 — Category-free inference (4 settings × seeds in this GPU) ─────────
run_p0_cell() {
    local SETTING=$1 SEED=$2
    local OUT="outputs/eval_uniform_mask/${SETTING}_s${SEED}.json"
    if [ -f "$OUT" ]; then
        echo "  [P0 ${SETTING} s${SEED}] DONE, skipping ($OUT)"
        return
    fi
    echo "  [P0 ${SETTING} s${SEED}] running ... ($(date))"
    $PYTHON scripts/eval_uniform_mask.py \
        --setting "$SETTING" --seed "$SEED" --device "$DEVICE" \
        2>&1 | tee "outputs/eval_uniform_mask/${SETTING}_s${SEED}.log" || \
        echo "  [P0 ${SETTING} s${SEED}] FAILED — check log"
    echo "  [P0 ${SETTING} s${SEED}] finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[seed=$SEED] P0 uniform-mask eval — 4 settings"
    # Order: CIFAR (cheap) first to validate pipeline, then ViT, NLP, LoRA
    run_p0_cell cifar "$SEED"
    run_p0_cell vit   "$SEED"
    run_p0_cell nlp   "$SEED"
    run_p0_cell lora  "$SEED"
done

# ─── P1 — Specialization diagnostics (s42 weights, heatmaps) ───────────────
# Default: s42 GPU runs both ViT + LoRA spec; others skip.
# Override (load-balance across GPUs):
#   FORCE_P1_VIT=1     run P1 ViT on this GPU regardless of SEED
#   FORCE_P1_LORA=1    run P1 LoRA on this GPU regardless of SEED
#   SKIP_P1_VIT=1      skip P1 ViT even on s42 GPU (so another GPU runs it)
#   SKIP_P1_LORA=1     skip P1 LoRA even on s42 GPU
# Diagnostics always operate on s42 weights (single-seed by design); the env
# var only controls WHICH physical GPU does the compute.
_HAS_S42=0
if [[ " ${SEEDS[*]} " =~ " 42 " ]]; then _HAS_S42=1; fi
_RUN_P1_VIT=0
if   [ "${FORCE_P1_VIT:-0}" = "1" ]; then _RUN_P1_VIT=1
elif [ "$_HAS_S42" = "1" ] && [ "${SKIP_P1_VIT:-0}" != "1" ]; then _RUN_P1_VIT=1
fi
_RUN_P1_LORA=0
if   [ "${FORCE_P1_LORA:-0}" = "1" ]; then _RUN_P1_LORA=1
elif [ "$_HAS_S42" = "1" ] && [ "${SKIP_P1_LORA:-0}" != "1" ]; then _RUN_P1_LORA=1
fi

if [ "$_RUN_P1_VIT" = "1" ] || [ "$_RUN_P1_LORA" = "1" ]; then
    echo ""
    echo "============================================================"
    echo " P1 specialization diagnostics (s42 weights — heatmaps)"
    echo "  RUN_P1_VIT=$_RUN_P1_VIT  RUN_P1_LORA=$_RUN_P1_LORA"
    echo "============================================================"

    # ViT
    if [ "$_RUN_P1_VIT" = "1" ]; then
        VIT_RUN_DIR="outputs/rtx5090_imagenet_vit_faithful/ours_vit_s42"
        VIT_OUT="outputs/analysis/vit_diag/ours_vit_s42.json"
        if [ -f "$VIT_OUT" ]; then
            echo "  [P1 ViT] DONE, skipping ($VIT_OUT)"
        elif [ -d "$VIT_RUN_DIR" ] && [ -f "$VIT_RUN_DIR/best.pt" ]; then
            echo "  [P1 ViT] running per-supercat × per-branch ablation ..."
            $PYTHON scripts/diagnose_vit_specialization.py \
                --run_dir "$VIT_RUN_DIR" --device "$DEVICE" --max_batches 200 \
                2>&1 | tee outputs/analysis/vit_diag/ours_vit_s42.log
        else
            echo "  [P1 ViT] SKIPPED — $VIT_RUN_DIR/best.pt not found (ours_vit not done?)"
        fi
    fi

    # LoRA
    if [ "$_RUN_P1_LORA" = "1" ]; then
        LORA_RUN_DIR="outputs/rtx5090_lora_faithful/ours_s42"
        LORA_OUT="outputs/analysis/lora_diag/ours_s42.json"
        if [ -f "$LORA_OUT" ]; then
            echo "  [P1 LoRA] DONE, skipping ($LORA_OUT)"
        elif [ -d "$LORA_RUN_DIR" ] && [ -f "$LORA_RUN_DIR/best.pt" ]; then
            echo "  [P1 LoRA] running per-cluster × per-branch ablation ..."
            $PYTHON scripts/diagnose_lora_specialization.py \
                --run_dir "$LORA_RUN_DIR" --device "$DEVICE" \
                2>&1 | tee outputs/analysis/lora_diag/ours_s42.log
        else
            echo "  [P1 LoRA] SKIPPED — $LORA_RUN_DIR/best.pt not found (ours not done?)"
        fi
    fi
fi

# ─── P1 LoRA SHARD MODE (parallel across 3 GPUs) ───────────────────────────
# Independent of the FULL-run P1 LoRA block above. Gated by env P1_LORA_SHARD
# ∈ {0, 1, 2}. Each GPU runs ~50min of branch evals; GPU 0 (shard 0) waits
# for shards 1 + 2 to land then runs the merger. Final output overwrites
# outputs/analysis/lora_diag/ours_s42.json.
#
# Default unset = no shard mode (existing FULL-run block above is the
# fallback path; backward compat preserved).
#
# Branch split (SuperNI K=20, 1 baseline + 20 ablations):
#   GPU 0 (shard 0): baseline + branches 0-6     (8 evals ≈ 49min wall)
#   GPU 1 (shard 1): branches 7-13               (7 evals ≈ 43min wall)
#   GPU 2 (shard 2): branches 14-19              (6 evals ≈ 37min wall)
# Saves ~70min vs sequential single-GPU full run.
if [ -n "${P1_LORA_SHARD:-}" ]; then
    SHARD=$P1_LORA_SHARD
    LORA_RUN_DIR="outputs/rtx5090_lora_faithful/ours_s42"
    SHARD_OUT="outputs/analysis/lora_diag/ours_s42.shard${SHARD}.json"
    FINAL_OUT="outputs/analysis/lora_diag/ours_s42.json"
    SHARD1_OUT="outputs/analysis/lora_diag/ours_s42.shard1.json"
    SHARD2_OUT="outputs/analysis/lora_diag/ours_s42.shard2.json"
    case "$SHARD" in
        0) BRANCHES="0,1,2,3,4,5,6"; SKIP_BASE="" ;;
        1) BRANCHES="7,8,9,10,11,12,13"; SKIP_BASE="--skip_baseline" ;;
        2) BRANCHES="14,15,16,17,18,19"; SKIP_BASE="--skip_baseline" ;;
        *) echo "ERROR: P1_LORA_SHARD=$SHARD invalid (must be 0|1|2)"; exit 1 ;;
    esac
    mkdir -p outputs/analysis/lora_diag

    echo ""
    echo "============================================================"
    echo " P1 LoRA SHARD $SHARD — branches $BRANCHES"
    echo "============================================================"
    if [ -f "$SHARD_OUT" ]; then
        echo "  [P1 LoRA shard $SHARD] DONE, skipping ($SHARD_OUT)"
    elif [ -d "$LORA_RUN_DIR" ] && [ -f "$LORA_RUN_DIR/best.pt" ]; then
        echo "  [P1 LoRA shard $SHARD] running ..."
        $PYTHON scripts/diagnose_lora_specialization.py \
            --run_dir "$LORA_RUN_DIR" --device "$DEVICE" \
            --branch_subset "$BRANCHES" $SKIP_BASE \
            --out_json "$SHARD_OUT" 2>&1 \
            | tee "outputs/analysis/lora_diag/shard${SHARD}.log"
    else
        echo "  [P1 LoRA shard $SHARD] SKIPPED — $LORA_RUN_DIR/best.pt missing"
    fi

    # GPU 0 (shard 0) merges after shards 1 + 2 land. Polls every 60s with
    # a 90-min ceiling so the chain doesn't hang if a sibling GPU dies.
    if [ "$SHARD" = "0" ]; then
        echo ""
        echo "  [P1 LoRA merge] waiting for shards 1 + 2 ..."
        WAIT_TIMEOUT=5400
        elapsed=0
        while [ $elapsed -lt $WAIT_TIMEOUT ]; do
            if [ -f "$SHARD1_OUT" ] && [ -f "$SHARD2_OUT" ]; then break; fi
            sleep 60
            elapsed=$((elapsed + 60))
            echo "    [merge wait] ${elapsed}s elapsed; "\
                 "shard1=$([ -f $SHARD1_OUT ] && echo OK || echo MISSING)  "\
                 "shard2=$([ -f $SHARD2_OUT ] && echo OK || echo MISSING)"
        done
        if [ -f "$SHARD1_OUT" ] && [ -f "$SHARD2_OUT" ]; then
            echo "  [P1 LoRA merge] all 3 shards present, merging ..."
            $PYTHON scripts/merge_lora_diag.py \
                "$SHARD_OUT" "$SHARD1_OUT" "$SHARD2_OUT" \
                --out "$FINAL_OUT" 2>&1 \
                | tee outputs/analysis/lora_diag/merge.log
        else
            echo "  [P1 LoRA merge] TIMEOUT — shards 1/2 still missing after 90min."
            echo "  Manual rescue: once shards 1/2 are present, run:"
            echo "    $PYTHON scripts/merge_lora_diag.py \\"
            echo "        $SHARD_OUT $SHARD1_OUT $SHARD2_OUT --out $FINAL_OUT"
        fi
    fi
fi

echo ""
echo "============================================================"
echo " rtx5090_10 complete: $(date)"
echo "============================================================"

# Per-GPU summary table
echo "P0 uniform-mask cells:"
for SEED in "${SEEDS[@]}"; do
    for SETTING in cifar vit nlp lora; do
        F="outputs/eval_uniform_mask/${SETTING}_s${SEED}.json"
        if [ -f "$F" ]; then
            $PYTHON -c "
import json
d = json.load(open('$F'))
print(f\"  {d['setting']:<6} s{d['seed']}: src={d['src_metric']:.4f}  uniform={d['uniform_mask_metric']:.4f}  Δ={d['delta']:+.4f}  ({d['metric_name']})\")
"
        fi
    done
done

echo ""
echo "P1 specialization diagnostics:"
# if-then-fi (not [ ] && cmd) so missing files don't bubble up rc=1 from
# the script's last command and trip set -e in the dispatch shell. Files
# may be missing on a given GPU (load-balanced across GPUs after 2026-05-01).
if [ -f "outputs/analysis/vit_diag/ours_vit_s42.json" ]; then
    $PYTHON -c "import json; d=json.load(open('outputs/analysis/vit_diag/ours_vit_s42.json')); print(f\"  ViT  : diag_hits={d['diag_hits']}/{d['n_cats']}  max|Δ|={d['max_abs_delta']:.2f} top1\")"
fi
if [ -f "outputs/analysis/lora_diag/ours_s42.json" ]; then
    $PYTHON -c "import json; d=json.load(open('outputs/analysis/lora_diag/ours_s42.json')); print(f\"  LoRA : diag_hits={d['diag_hits']}/{d['n_cats']}  max|Δ|={d['max_abs_delta']:.4f} ROUGE-L\")"
fi
