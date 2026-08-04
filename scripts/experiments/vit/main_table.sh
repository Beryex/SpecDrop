#!/bin/bash
###############################################################################
# ImageNet-1K × ViT-Small, faithful main table.
#
# Final ours config is chosen by the 5a → 5b → 5c ablation chain, which
# writes _best_{pa,beta,se}.txt markers that this script reads. Without the
# markers it stops with instructions; to skip the chain, export the paper's
# final operating point directly: BEST_PA=0.6 BEST_BETA=1 BEST_SE=2.0.
#
# METHOD COMPONENTS (all always-on for ours, matching paper main body):
#   • Uniform K=46 branches (BREEDS superclass mapping)
#   • Per-category routing: frac_per_category + amplification_beta=β (best_β)
#   • pa warmup: cosine schedule, warmup_unit='epoch' (matches LR granularity —
#     trainer.py cosine LR is per-epoch, so pa warmup is per-epoch too per
#     the paper's warmup-granularity-alignment design principle)
#   • Shared-expert FFN: shared_expert_dim > 0, per split_ffn_budget_for_se
#   • pa=0.5 excluded a priori from argmax (mechanism-OFF degenerate point)
#
# Baselines (faithful to each MoE method's native ViT paper):
#   1. ViT-Small/16 (dense, reference, 22.05M params)
#   2. Mod-Squad ViT (Chen CVPR 2023) — NATIVE ViT MoE
#   3. Soft MoE ViT (Puigcerver ICLR 2024) — NATIVE ViT slot routing
#   4. COMET ViT (Shaier ICLR 2025) — NATIVE ViT fixed-random k-WTA
#   5. MultiBranchViT-S/16 no-routing (K=46, no SE — architectural reference)
#   6. MultiBranchViT-S/16 no-routing + SE=best_SE (matched-SE scalar baseline
#      — directly comparable to ours; pa=0.5 of our method is mathematically
#      equivalent to this by the same argument used in the NLP 500M table)
#   7. SpecDrop ViT (OURS) — K=46, pa=best_pa β=best_β SE=best_SE epoch-cosine
#
# Setting (DeiT training recipe, short-recipe variant; matches all baselines
# + ours identically — see COMMON_TRAINING below):
#   Optimizer: AdamW lr=2.5e-4 (DeiT linear-scaling rule: 5e-4 × batch/512;
#              at batch=256 → 2.5e-4), weight_decay=0.05, β1=0.9, β2=0.999
#   Schedule:  Linear warmup (5 ep) + cosine annealing (95 ep) — per-epoch
#   Epochs:    100, Batch: 256 (grad_accum 1), Seeds: 42 / 123 / 456
#   Aug:       DeiT (RandAug 2,9 + RandomErasing 0.25), Mixup 0.8 + CutMix 1.0
#   Label smoothing 0.1, drop_path_rate 0.1, bf16 AMP, SDPA (flash-attn-2 kernel)
# Intentionally omitted vs DeiT canonical 300-ep recipe: EMA (τ=0.9999) and
# repeated augmentation ×3. All methods share this short-recipe for
# apples-to-apples comparison; paper discloses in Experimental Setup.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_imagenet_vit_faithful"
MINI_BASE=${MINI_BASE:-"./outputs/rtx5090_vit_mini_ablation"}
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# Final ours config — auto-read from 5a/5b/5c markers written at the end of
# the ablation chain:
#   $MINI_BASE/_best_pa.txt    (written by rtx5090_5b)
#   $MINI_BASE/_best_beta.txt  (written by rtx5090_5c after 5b barrier)
#   $MINI_BASE/_best_se.txt    (written by rtx5090_5c after its own barrier)
# Env override hierarchy:  explicit env > marker file > fail loud.
# Do NOT hardcode placeholder defaults — that risks shipping wrong numbers if
# the ablation fails silently.

_read_marker() {
    local var_name=$1 marker=$2 label=$3
    if [ -n "${!var_name}" ]; then
        echo "[rtx5090_6] $label=${!var_name} (from env)"
        return
    fi
    if [ -f "$MINI_BASE/$marker" ]; then
        local v=$(cat "$MINI_BASE/$marker" | tr -d '[:space:]')
        if [ -n "$v" ]; then
            eval "$var_name=$v"
            echo "[rtx5090_6] $label=$v (from $MINI_BASE/$marker)"
            return
        fi
    fi
    echo "ERROR: $label not set. Either export $var_name=... or run the"
    echo "5a→5b→5c ablation chain first (it writes $MINI_BASE/$marker)."
    exit 1
}

_read_marker BEST_PA   _best_pa.txt   BEST_PA
_read_marker BEST_BETA _best_beta.txt BEST_BETA
_read_marker BEST_SE   _best_se.txt   BEST_SE
BEST_PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " ImageNet × ViT-S/16 FAITHFUL (7 methods × ${#SEEDS[@]} seeds)"
echo " OURS lock-in: pa=$BEST_PA β=$BEST_BETA SE=$BEST_SE epoch-cosine K=46"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 100, 'batch_size': 256, 'grad_accum_steps': 1, 'lr': 2.5e-4, 'optimizer': 'adamw', 'weight_decay': 0.05, 'lr_schedule': 'cosine', 'warmup_epochs': 5, 'label_smoothing': 0.1, 'mixup_alpha': 0.8, 'cutmix_alpha': 1.0, 'mixup_switch_prob': 0.5, 'amp_dtype': 'bf16', '_compile_mode': 'reduce-overhead', 'max_grad_norm': 1.0"
COMMON_DATA="'dataset': 'imagenet', 'data_dir': './data_cache/imagenet', 'num_workers': 16, 'prefetch_factor': 4, 'augmentation': 'deit'"

run_experiment() {
    local NAME=$1 CONFIG_GEN=$2
    for SEED in "${SEEDS[@]}"; do
        local ENAME="${NAME}_s${SEED}"
        local ODIR="${OUTDIR_BASE}/${ENAME}"
        if [ -f "$ODIR/results.json" ]; then
            echo "  $ENAME — DONE, skipping"; continue
        fi
        mkdir -p "$ODIR"
        echo "  Running $ENAME ... ($(date))"
        $PYTHON -c "
import yaml
$CONFIG_GEN
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
yaml.dump(cfg, open('$OUTDIR_BASE/_tmp_s${SEED}.yaml', 'w'))
"
        $PYTHON run.py --wandb --config "$OUTDIR_BASE/_tmp_s${SEED}.yaml" --device $DEVICE 2>&1 | tee "${OUTDIR_BASE}/${ENAME}.log"
        echo "  $ENAME finished: $(date)"
    done
}

echo "[1/7] ViT-Small/16 (dense, reference)"
run_experiment "vit_small" "
cfg = {
    'model': {'type': 'vit_small', 'num_classes': 1000, 'img_size': 224, 'patch_size': 16, 'drop_path_rate': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[2/7] Mod-Squad ViT (N=16, top_k=2, MI w=0.001)"
run_experiment "mod_squad_vit" "
cfg = {
    'model': {'type': 'mod_squad_vit', 'num_classes': 1000, 'num_experts': 16, 'top_k': 2, 'mi_weight': 0.001, 'drop_path_rate': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[3/7] Soft MoE ViT (N=32, slots=1, fully soft)"
run_experiment "soft_moe_vit" "
cfg = {
    'model': {'type': 'soft_moe_vit', 'num_classes': 1000, 'num_experts': 32, 'slots_per_expert': 1, 'drop_path_rate': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[4/7] COMET ViT (p_keep=0.25, fixed-random V buffers)"
run_experiment "comet_vit" "
cfg = {
    'model': {'type': 'comet_vit', 'num_classes': 1000, 'p_keep': 0.25, 'drop_path_rate': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

# Order rationale (changed 2026-04-29): K-dependent no-routing baselines run
# AFTER ours, NOT before. If ours @ K=46 underperforms dense (76.38) and we
# decide to retry with smaller K, the K-dependent no_routing+(SE) baselines
# would have to be re-run anyway. Putting them AFTER ours saves ~28h × 2 = 56h
# of compute that would otherwise be sunk on the wrong K. Independent baselines
# (vit_small / mod_squad / soft_moe / comet) are K-independent and stay first.

echo "[5/7] SpecDrop ViT (OURS) — K=46, pa=$BEST_PA β=$BEST_BETA SE=$BEST_SE epoch-cosine"
run_experiment "ours_vit" "
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se
fracs = compute_category_fractions('./data_cache/imagenet', NUM_SUPERCLASSES)
total_branch, se_dim = split_ffn_budget_for_se(384*4, $BEST_SE, 46)
branch_hidden = total_branch // 46
cfg = {
    'model': {'type': 'multi_branch_vit_small', 'num_classes': 1000, 'num_branches': 46,
               'branch_hidden': branch_hidden, 'shared_expert_dim': se_dim,
               'drop_path_rate': 0.1},
    'algorithm': {'type': 'soft_specdrop', 'p_active': $BEST_PA, 'p_inactive': $BEST_PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'warmup_schedule': 'cosine', 'warmup_unit': 'epoch',
                   'frac_per_category': fracs, 'amplification_beta': $BEST_BETA},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[6/7] MultiBranchViT no-routing (K=46, no SE — architectural ref)"
run_experiment "mbvit_no_routing" "
cfg = {
    'model': {'type': 'multi_branch_vit_small', 'num_classes': 1000, 'num_branches': 46, 'drop_path_rate': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[7/7] MultiBranchViT no-routing + SE=$BEST_SE (matched-SE scalar baseline)"
run_experiment "mbvit_no_routing_se" "
from data.slimpajama import split_ffn_budget_for_se
total_branch, se_dim = split_ffn_budget_for_se(384*4, $BEST_SE, 46)
branch_hidden = total_branch // 46
cfg = {
    'model': {'type': 'multi_branch_vit_small', 'num_classes': 1000, 'num_branches': 46,
               'branch_hidden': branch_hidden, 'shared_expert_dim': se_dim,
               'drop_path_rate': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo ""
echo "============================================================"
echo " ViT ImageNet FAITHFUL Results (~22M params)"
echo "============================================================"
$PYTHON -c "
import json, os
methods = ['vit_small','mod_squad_vit','soft_moe_vit','comet_vit',
           'mbvit_no_routing','mbvit_no_routing_se','ours_vit']
labels  = ['ViT-S','Mod-Squad','Soft MoE','COMET',
           'MB-ViT','MB-ViT+SE','Ours']
for m, l in zip(methods, labels):
    accs = []
    for s in [42, 123, 456]:
        path = f'$OUTDIR_BASE/{m}_s{s}/results.json'
        if os.path.exists(path):
            accs.append(json.load(open(path))['best_top1'])
    if accs:
        mean = sum(accs)/len(accs)
        std = (sum((x-mean)**2 for x in accs)/len(accs))**0.5
        print(f'  {l:15s}: {mean:.2f} +/- {std:.2f}  (n={len(accs)})')
"
echo "Done: $(date)"
