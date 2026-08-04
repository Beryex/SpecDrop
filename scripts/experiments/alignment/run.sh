#!/bin/bash
# rtx5090_15 — Branch-category alignment (pruning sensitivity) for ALL multibranch
# methods × 3 seeds across 4 settings: CIFAR / ViT / NLP / LoRA.
#
# Per-cell skip via output JSON existence guard. Each cell calls one of:
#   scripts/diagnose_{cifar,vit,nlp,lora}_specialization.py
# with --run_dir + --out_json.
#
# Distribution across 6 GPUs is via `_run_gpu{0..5}.sh` env vars selecting cell
# IDs. This script accepts:
#   CELLS_FILTER="<comma-separated-ids>"   restrict to a subset of cells.
#   Default = no var → run ALL cells (NOT recommended; use _run_gpu*.sh).
#
# Cell IDs are stable; see CELL_TABLE below.

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
set -e

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}

mkdir -p outputs/analysis/specialization
mkdir -p outputs/analysis/vit_diag
mkdir -p outputs/analysis/nlp_diag
mkdir -p outputs/analysis/lora_diag

# ─── Cell table ─────────────────────────────────────────────────────────────
# Each row: <cell_id>:<setting>:<run_dir>:<out_json>
# wall-time estimates (per-cell): CIFAR ~2min | ViT ~30min | NLP ~10min | LoRA ~50min
declare -a CELLS=(
  # CIFAR (9 cells, ~2 min each)
  "C04:cifar:outputs/rtx5090_cifar100_faithful/ours_s42:outputs/analysis/specialization/ours_s42.json"
  "C05:cifar:outputs/rtx5090_cifar100_faithful/ours_s123:outputs/analysis/specialization/ours_s123.json"
  "C06:cifar:outputs/rtx5090_cifar100_faithful/ours_s456:outputs/analysis/specialization/ours_s456.json"
  "C07:cifar:outputs/rtx5090_cifar100_faithful/no_routing_s42:outputs/analysis/specialization/no_routing_s42.json"
  "C08:cifar:outputs/rtx5090_cifar100_faithful/no_routing_s123:outputs/analysis/specialization/no_routing_s123.json"
  "C09:cifar:outputs/rtx5090_cifar100_faithful/no_routing_s456:outputs/analysis/specialization/no_routing_s456.json"
  "C01:cifar:outputs/cv_hard_category_k20/s42:outputs/analysis/specialization/hard_category_s42.json"
  "C02:cifar:outputs/cv_hard_category_k20/s123:outputs/analysis/specialization/hard_category_s123.json"
  "C03:cifar:outputs/cv_hard_category_k20/s456:outputs/analysis/specialization/hard_category_s456.json"

  # ViT (9 cells, ~30 min each)
  "V15:vit:outputs/rtx5090_imagenet_vit_faithful/ours_vit_s42:outputs/analysis/vit_diag/ours_vit_s42.json"
  "V01:vit:outputs/rtx5090_imagenet_vit_faithful/ours_vit_s123:outputs/analysis/vit_diag/ours_vit_s123.json"
  "V02:vit:outputs/rtx5090_imagenet_vit_faithful/ours_vit_s456:outputs/analysis/vit_diag/ours_vit_s456.json"
  "V03:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_s42:outputs/analysis/vit_diag/mbvit_no_routing_s42.json"
  "V04:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_s123:outputs/analysis/vit_diag/mbvit_no_routing_s123.json"
  "V05:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_s456:outputs/analysis/vit_diag/mbvit_no_routing_s456.json"
  "V06:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_se_s42:outputs/analysis/vit_diag/mbvit_no_routing_se_s42.json"
  "V07:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_se_s123:outputs/analysis/vit_diag/mbvit_no_routing_se_s123.json"
  "V08:vit:outputs/rtx5090_imagenet_vit_faithful/mbvit_no_routing_se_s456:outputs/analysis/vit_diag/mbvit_no_routing_se_s456.json"

  # NLP (12 cells, ~10 min each)
  "N23:nlp:outputs/rtx5090_nlp_faithful/ours_phaseP_s42:outputs/analysis/nlp_diag/ours_phaseP_s42.json"
  "N24:nlp:outputs/rtx5090_nlp_faithful/no_routing_se05_s42:outputs/analysis/nlp_diag/no_routing_se05_s42.json"
  "N01:nlp:outputs/rtx5090_nlp_faithful/ours_phaseP_s123:outputs/analysis/nlp_diag/ours_phaseP_s123.json"
  "N02:nlp:outputs/rtx5090_nlp_faithful/ours_phaseP_s456:outputs/analysis/nlp_diag/ours_phaseP_s456.json"
  "N03:nlp:outputs/rtx5090_nlp_faithful/no_routing_s42:outputs/analysis/nlp_diag/no_routing_s42.json"
  "N04:nlp:outputs/rtx5090_nlp_faithful/no_routing_s123:outputs/analysis/nlp_diag/no_routing_s123.json"
  "N05:nlp:outputs/rtx5090_nlp_faithful/no_routing_s456:outputs/analysis/nlp_diag/no_routing_s456.json"
  "N06:nlp:outputs/rtx5090_nlp_faithful/no_routing_se_s42:outputs/analysis/nlp_diag/no_routing_se_s42.json"
  "N07:nlp:outputs/rtx5090_nlp_faithful/no_routing_se_s123:outputs/analysis/nlp_diag/no_routing_se_s123.json"
  "N08:nlp:outputs/rtx5090_nlp_faithful/no_routing_se_s456:outputs/analysis/nlp_diag/no_routing_se_s456.json"
  "N09:nlp:outputs/rtx5090_nlp_faithful/no_routing_se05_s123:outputs/analysis/nlp_diag/no_routing_se05_s123.json"
  "N10:nlp:outputs/rtx5090_nlp_faithful/no_routing_se05_s456:outputs/analysis/nlp_diag/no_routing_se05_s456.json"

  # LoRA SpecDrop family (9 cells, ~50 min each unsharded)
  "L17:lora:outputs/rtx5090_lora_faithful/ours_s42:outputs/analysis/lora_diag/ours_s42.json"
  "L18:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_s42:outputs/analysis/lora_diag/mb_lora_no_routing_s42.json"
  "L01:lora:outputs/rtx5090_lora_faithful/ours_s123:outputs/analysis/lora_diag/ours_s123.json"
  "L02:lora:outputs/rtx5090_lora_faithful/ours_s456:outputs/analysis/lora_diag/ours_s456.json"
  "L03:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_s123:outputs/analysis/lora_diag/mb_lora_no_routing_s123.json"
  "L04:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_s456:outputs/analysis/lora_diag/mb_lora_no_routing_s456.json"
  "L05:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_se1.0_s42:outputs/analysis/lora_diag/mb_lora_no_routing_se1.0_s42.json"
  "L06:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_se1.0_s123:outputs/analysis/lora_diag/mb_lora_no_routing_se1.0_s123.json"
  "L07:lora:outputs/rtx5090_lora_faithful/mb_lora_no_routing_se1.0_s456:outputs/analysis/lora_diag/mb_lora_no_routing_se1.0_s456.json"

  # LoRA learned routers (9 cells, ~50 min each — uses ablation hooks per scripts/_ablation_hooks.py)
  "L08:lora:outputs/rtx5090_lora_faithful/hydra_lora_n8_s42:outputs/analysis/lora_diag/hydra_lora_n8_s42.json"
  "L09:lora:outputs/rtx5090_lora_faithful/hydra_lora_n8_s123:outputs/analysis/lora_diag/hydra_lora_n8_s123.json"
  "L10:lora:outputs/rtx5090_lora_faithful/hydra_lora_n8_s456:outputs/analysis/lora_diag/hydra_lora_n8_s456.json"
  "L11:lora:outputs/rtx5090_lora_faithful/loramoe_k6_s42:outputs/analysis/lora_diag/loramoe_k6_s42.json"
  "L12:lora:outputs/rtx5090_lora_faithful/loramoe_k6_s123:outputs/analysis/lora_diag/loramoe_k6_s123.json"
  "L13:lora:outputs/rtx5090_lora_faithful/loramoe_k6_s456:outputs/analysis/lora_diag/loramoe_k6_s456.json"
  "L14:lora:outputs/rtx5090_lora_faithful/mocle_e4_s42:outputs/analysis/lora_diag/mocle_e4_s42.json"
  "L15:lora:outputs/rtx5090_lora_faithful/mocle_e4_s123:outputs/analysis/lora_diag/mocle_e4_s123.json"
  "L16:lora:outputs/rtx5090_lora_faithful/mocle_e4_s456:outputs/analysis/lora_diag/mocle_e4_s456.json"

  # ViT learned routers (6 cells, ~30 min each — uses ablation hooks)
  "V09:vit:outputs/rtx5090_imagenet_vit_faithful/soft_moe_vit_s42:outputs/analysis/vit_diag/soft_moe_vit_s42.json"
  "V10:vit:outputs/rtx5090_imagenet_vit_faithful/soft_moe_vit_s123:outputs/analysis/vit_diag/soft_moe_vit_s123.json"
  "V11:vit:outputs/rtx5090_imagenet_vit_faithful/soft_moe_vit_s456:outputs/analysis/vit_diag/soft_moe_vit_s456.json"
  "V12:vit:outputs/rtx5090_imagenet_vit_faithful/mod_squad_vit_s42:outputs/analysis/vit_diag/mod_squad_vit_s42.json"
  "V13:vit:outputs/rtx5090_imagenet_vit_faithful/mod_squad_vit_s123:outputs/analysis/vit_diag/mod_squad_vit_s123.json"
  "V14:vit:outputs/rtx5090_imagenet_vit_faithful/mod_squad_vit_s456:outputs/analysis/vit_diag/mod_squad_vit_s456.json"

  # NLP native MoE (12 cells, ~10 min each — uses ablation hooks via ParallelFFN post-hook)
  "N11:nlp:outputs/rtx5090_nlp_faithful/switch_s42:outputs/analysis/nlp_diag/switch_s42.json"
  "N12:nlp:outputs/rtx5090_nlp_faithful/switch_s123:outputs/analysis/nlp_diag/switch_s123.json"
  "N13:nlp:outputs/rtx5090_nlp_faithful/switch_s456:outputs/analysis/nlp_diag/switch_s456.json"
  "N14:nlp:outputs/rtx5090_nlp_faithful/hash_layers_s42:outputs/analysis/nlp_diag/hash_layers_s42.json"
  "N15:nlp:outputs/rtx5090_nlp_faithful/hash_layers_s123:outputs/analysis/nlp_diag/hash_layers_s123.json"
  "N16:nlp:outputs/rtx5090_nlp_faithful/hash_layers_s456:outputs/analysis/nlp_diag/hash_layers_s456.json"
  "N17:nlp:outputs/rtx5090_nlp_faithful/demix_s42:outputs/analysis/nlp_diag/demix_s42.json"
  "N18:nlp:outputs/rtx5090_nlp_faithful/demix_s123:outputs/analysis/nlp_diag/demix_s123.json"
  "N19:nlp:outputs/rtx5090_nlp_faithful/demix_s456:outputs/analysis/nlp_diag/demix_s456.json"
  "N20:nlp:outputs/rtx5090_nlp_faithful/smoe_dropout_s42:outputs/analysis/nlp_diag/smoe_dropout_s42.json"
  "N21:nlp:outputs/rtx5090_nlp_faithful/smoe_dropout_s123:outputs/analysis/nlp_diag/smoe_dropout_s123.json"
  "N22:nlp:outputs/rtx5090_nlp_faithful/smoe_dropout_s456:outputs/analysis/nlp_diag/smoe_dropout_s456.json"
)

# Optional CELLS_FILTER env var subsets the run.
if [ -n "${CELLS_FILTER}" ]; then
    IFS=',' read -ra wanted <<< "${CELLS_FILTER}"
    declare -a SELECTED=()
    for cell in "${CELLS[@]}"; do
        cid="${cell%%:*}"
        for w in "${wanted[@]}"; do
            if [ "$cid" = "$w" ]; then
                SELECTED+=("$cell")
                break
            fi
        done
    done
    CELLS=("${SELECTED[@]}")
fi

if [ ${#CELLS[@]} -eq 0 ]; then
    echo "[rtx5090_15] no cells selected (CELLS_FILTER='${CELLS_FILTER}'); exiting"
    exit 0
fi

echo "============================================================"
echo " rtx5090_15: alignment / pruning sensitivity"
echo "   cells: ${#CELLS[@]}  date: $(date)"
for c in "${CELLS[@]}"; do echo "   - $c"; done
echo "============================================================"

run_cell() {
    local row="$1"
    local cid="${row%%:*}"
    local rest="${row#*:}"
    local setting="${rest%%:*}"
    local rest2="${rest#*:}"
    local run_dir="${rest2%%:*}"
    local out_json="${rest2#*:}"

    if [ -f "$out_json" ]; then
        echo "[$cid] cached: $out_json (skipping)"
        return 0
    fi
    if [ ! -d "$run_dir" ]; then
        echo "[$cid] MISSING run_dir: $run_dir" >&2
        return 1
    fi

    echo "[$cid] $(date +%H:%M:%S) START $setting $run_dir"
    case "$setting" in
        cifar)
            $PYTHON scripts/diagnose_cifar_specialization.py \
                --run_dir "$run_dir" --out_json "$out_json" --device "$DEVICE"
            ;;
        vit)
            $PYTHON scripts/diagnose_vit_specialization.py \
                --run_dir "$run_dir" --out_json "$out_json" --device "$DEVICE"
            ;;
        nlp)
            $PYTHON scripts/diagnose_nlp_specialization.py \
                --run_dir "$run_dir" --out_json "$out_json" --device "$DEVICE"
            ;;
        lora)
            # LoRA: full unsharded run (single GPU does all K branches).
            # If you want to shard further, see scripts/diagnose_lora_specialization.py
            # --branch_subset / --skip_baseline + scripts/merge_lora_diag.py.
            $PYTHON scripts/diagnose_lora_specialization.py \
                --run_dir "$run_dir" --out_json "$out_json" --device "$DEVICE"
            ;;
        *)
            echo "[$cid] unknown setting: $setting" >&2
            return 1
            ;;
    esac
    echo "[$cid] $(date +%H:%M:%S) DONE"
}

for cell in "${CELLS[@]}"; do
    run_cell "$cell" || echo "[rtx5090_15] cell ${cell%%:*} FAILED, continuing"
done

echo "[rtx5090_15] all cells processed at $(date)"
