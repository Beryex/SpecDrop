#!/bin/bash
# reproduce.sh — single entry-point for paper reproduction.
#
# Usage:
#   bash reproduce.sh <target>
#
# Targets:
#   smoke         Quick environment + pipeline validation (all 4 settings; ~30-50 min on a
#                 first run including one-time dataset downloads, ~10 min warm).
#   cifar         Table 1: CIFAR-100, 7 methods × 3 seeds × 200 epochs (~18 GPU-hours).
#   nlp           Table 3: SlimPajama 30M Transformer, 8 table methods + 2 provenance
#                 configs × 3 seeds × 10 epochs
#                 × 500M tokens (~150 GPU-hours; SlimPajama tokenize is one-time ~30 min).
#   vit           Table 2: ImageNet ViT-S/16 K=46 BREEDS, all 9 table rows × 3 seeds × 100 epochs
#                 (~630 GPU-hours; ImageNet-1K download is one-time, ~150 GB).
#   lora          Table 4: SuperNI Llama-3.2-1B + LoRA, 7 methods × 3 seeds × 3 epochs
#                 (~290 GPU-hours; HF Llama download + SuperNI tasks are one-time).
#   alignment     Align columns of Tables 1-4: branch-category alignment (~24 GPU-hours
#                 single-GPU; uses pre-trained checkpoints from cifar/nlp/vit/lora targets).
#   ablation_<setting>  Run only the (pa, β, SE) sweep ablations for one setting.
#                       e.g. `bash reproduce.sh ablation_nlp`.
#   all           cifar + nlp + vit + lora + alignment (≈1100 GPU-hours; multi-GPU strongly
#                 recommended; see README §Reproducibility).
#
# Hardware:
#   - Single-GPU: 1× RTX 5090 (or any GPU with ≥24 GB VRAM and bf16 support).
#   - Multi-GPU parallel (per-seed): launch one process per GPU with
#       CUDA_VISIBLE_DEVICES=<i> SEEDS_OVERRIDE=<seed> bash <chain>.sh
#     example for CIFAR with 3 GPUs:
#       CUDA_VISIBLE_DEVICES=0 SEEDS_OVERRIDE=42  bash scripts/experiments/cifar/main_table.sh &
#       CUDA_VISIBLE_DEVICES=1 SEEDS_OVERRIDE=123 bash scripts/experiments/cifar/main_table.sh &
#       CUDA_VISIBLE_DEVICES=2 SEEDS_OVERRIDE=456 bash scripts/experiments/cifar/main_table.sh &
#       wait
#
# Environment:
#   conda activate SpecDrop  (or any env that satisfies requirements.txt)
#   Repository root must be the current working directory.
#
# Caches (auto-created on first run):
#   data_cache/   tokenized SlimPajama, ImageNet meta, SuperNI tasks (gitignored)
#   outputs/      per-cell results.json + best.pt (gitignored)
#   wandb/        optional W&B run logs (gitignored)
set -e

if [ "$#" -lt 1 ]; then
    sed -n '2,40p' "$0"
    exit 1
fi

TARGET="$1"
shift

run() {
    echo "============================================================"
    echo " reproduce.sh → bash $*"
    echo "============================================================"
    bash "$@"
}

case "$TARGET" in
    smoke)
        run scripts/experiments/smoke/cifar.sh
        run scripts/experiments/smoke/nlp.sh
        run scripts/experiments/smoke/vit.sh
        run scripts/experiments/smoke/lora.sh
        ;;

    cifar)
        # Paper operating point for the CIFAR SE sweep (the pa/wr sweeps write no
        # marker files; cifar/main_table.sh hardcodes pa=0.7, wr=1.0 regardless).
        export BEST_PA=${BEST_PA:-0.7} BEST_WR=${BEST_WR:-1.0}
        run scripts/experiments/cifar/ablation_pa.sh
        run scripts/experiments/cifar/ablation_wr.sh
        run scripts/experiments/cifar/ablation_se.sh
        run scripts/experiments/cifar/main_table.sh
        ;;
    ablation_cifar)
        export BEST_PA=${BEST_PA:-0.7} BEST_WR=${BEST_WR:-1.0}
        run scripts/experiments/cifar/ablation_pa.sh
        run scripts/experiments/cifar/ablation_wr.sh
        run scripts/experiments/cifar/ablation_se.sh
        ;;

    nlp)
        run scripts/experiments/nlp/ablation_pa.sh
        run scripts/experiments/nlp/ablation_beta.sh
        run scripts/experiments/nlp/ablation_se.sh
        run scripts/experiments/nlp/ablation_step_warmup.sh
        run scripts/experiments/nlp/main_table.sh
        ;;
    ablation_nlp)
        run scripts/experiments/nlp/ablation_pa.sh
        run scripts/experiments/nlp/ablation_beta.sh
        run scripts/experiments/nlp/ablation_se.sh
        run scripts/experiments/nlp/ablation_step_warmup.sh
        ;;

    vit)
        run scripts/experiments/vit/ablation_pa.sh
        run scripts/experiments/vit/ablation_beta.sh
        run scripts/experiments/vit/ablation_se.sh
        run scripts/experiments/vit/main_table.sh
        # Remaining Table-2 rows: tuned Soft MoE (r7), compute-matched Soft MoE (r6),
        # and the auxiliary-loss-free top-k router — full protocol, 3 seeds each.
        FOREGROUND=1 run scripts/experiments/softmoe_study_finals.sh softmoe_study_r7
        FOREGROUND=1 run scripts/experiments/softmoe_study_finals.sh softmoe_study_r6
        FOREGROUND=1 run scripts/experiments/softmoe_study_finals.sh alf_moe_vit
        ;;
    ablation_vit)
        run scripts/experiments/vit/ablation_pa.sh
        run scripts/experiments/vit/ablation_beta.sh
        run scripts/experiments/vit/ablation_se.sh
        ;;

    lora)
        run scripts/experiments/lora/ablation_pa.sh
        run scripts/experiments/lora/ablation_beta.sh
        run scripts/experiments/lora/ablation_se.sh
        run scripts/experiments/lora/main_table.sh
        ;;
    ablation_lora)
        run scripts/experiments/lora/ablation_pa.sh
        run scripts/experiments/lora/ablation_beta.sh
        run scripts/experiments/lora/ablation_se.sh
        ;;

    alignment)
        # Requires trained checkpoints from cifar/nlp/vit/lora main_table targets.
        # Single-GPU: ~24 hours. For 6-GPU parallel, launch the per-GPU shards
        # under scripts/experiments/alignment/_run_gpu{0..5}.sh in parallel
        # (see README §Reproducibility).
        run scripts/experiments/alignment/run.sh
        echo
        echo "Aggregate per-method × per-seed alignment fractions:"
        echo "  python scripts/aggregate_alignment.py"
        ;;

    all)
        bash "$0" cifar
        bash "$0" nlp
        bash "$0" vit
        bash "$0" lora
        bash "$0" alignment
        ;;

    -h|--help|help)
        sed -n '2,40p' "$0"
        ;;

    *)
        echo "Unknown target: $TARGET" >&2
        echo "Run 'bash reproduce.sh --help' for usage." >&2
        exit 1
        ;;
esac
