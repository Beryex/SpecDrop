#!/bin/bash
###############################################################################
# RTX 5090 Script 14: E-NLP-1 — bs=64 sanity at 30M (W asymmetric-tuning
# check + bs=32 silent-override remediation).
#
# Two cells × 1 seed (s42):
#   [7]  ours_phaseP_bs64   pa=0.6 β=4 SE=0.5 step warmup, batch_size=64
#   [8]  no_routing_se05_bs64  arch-matched scalar (NoDropout uniform 1/K), bs=64
#
# Why: paper-locked 30M numbers were trained at effective bs=32 (silent
# override bug, since fixed). The bs=64 sanity check confirms
# rankings unchanged. data.batch_size AND training.batch_size both set to
# 64 explicitly (defensive against any cfg-path regression).
#
# Output dir: ./outputs/rtx5090_nlp_bs64/ — separate from paper-locked
# `rtx5090_nlp_faithful/` so existing results remain untouched.
#
# Per-cell wall ~8h on RTX 5090 (10 epochs × 500M tokens × bs=64). 2 cells
# run in parallel on GPU 0 + GPU 1 from _run_gpu0.sh / _run_gpu1.sh.
###############################################################################

set -eo pipefail
PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_bs64"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42); fi

# E-NLP-1 spec: only run cells listed in METHODS_OVERRIDE (default: both).
if [ -n "$METHODS_OVERRIDE" ]; then
    METHODS_TO_RUN=($METHODS_OVERRIDE)
else
    METHODS_TO_RUN=(ours_phaseP no_routing_se05)
fi
method_enabled() {
    local m=$1
    for x in "${METHODS_TO_RUN[@]}"; do
        if [ "$x" = "$m" ]; then return 0; fi
    done
    return 1
}

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " E-NLP-1: bs=64 30M sanity — seeds: ${SEEDS[*]}  methods: ${METHODS_TO_RUN[*]}"
echo " $(date)"
echo "============================================================"

# Identical to rtx5090_4 paper-canonical 30M config EXCEPT bs=32→64 (under
# both training: and data: as defensive). 10 epochs × 500M tokens × bs=64.
COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'max_tokens': 500_000_000, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'batch_size': 64"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"

run_experiment() {
    local NAME=$1 CONFIG_GEN=$2
    for SEED in "${SEEDS[@]}"; do
        local ENAME="${NAME}_s${SEED}"
        local ODIR="${OUTDIR_BASE}/${ENAME}"
        if [ -f "$ODIR/results.json" ]; then
            echo "  $ENAME — DONE, skipping"; continue
        fi
        mkdir -p "$ODIR"
        local YAML="$OUTDIR_BASE/_tmp_${NAME}_s${SEED}.yaml"
        rm -f "$YAML"
        echo "  Running $ENAME ... ($(date))"
        if ! $PYTHON -c "
import yaml
$CONFIG_GEN
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
yaml.dump(cfg, open('$YAML', 'w'))
"; then
            echo "  ERROR: config-gen failed for $ENAME. Aborting cell."
            return 1
        fi
        if [ ! -f "$YAML" ]; then
            echo "  ERROR: $YAML missing after config-gen. Aborting cell."
            return 1
        fi
        $PYTHON run_nlp.py --wandb --config "$YAML" --device $DEVICE 2>&1 | tee "${OUTDIR_BASE}/${ENAME}.log"
        echo "  $ENAME finished: $(date)"
    done
}

if method_enabled "ours_phaseP"; then
echo "[1/2] Soft SpecDrop (OURS, paper-canonical Phase P at bs=64)"
echo "       frac_per_category auto-loaded from 500M train tokenize cache"
run_experiment "ours_phaseP" "
from data.slimpajama import find_tokenize_cache, compute_category_fractions, split_ffn_budget_for_se
cache = find_tokenize_cache(data_dir='./data_cache/slimpajama', max_tokens=500_000_000, max_seq_len=512)
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[ours_phaseP_bs64] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={7*ffn_per_branch + se_dim}')
print(f'[ours_phaseP_bs64] frac_per_category (from 500M cache): {[round(f, 4) for f in fracs]}')
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED,
              'num_branches': 7, 'ffn_dim_per_branch': ffn_per_branch,
              'shared_expert_dim': se_dim, 'dropout': 0.1},
    'algorithm': {'type': 'soft_specdrop', 'p_active': 0.6, 'p_inactive': 0.4,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'warmup_schedule': 'cosine', 'warmup_unit': 'step',
                   'frac_per_category': fracs, 'amplification_beta': 4.0},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"
fi

if method_enabled "no_routing_se05"; then
echo "[2/2] MultiBranchTransformerLM no-routing + SE=0.5 (matched scalar at bs=64)"
run_experiment "no_routing_se05" "
from data.slimpajama import split_ffn_budget_for_se
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[no_routing_se05_bs64] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}')
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED,
              'num_branches': 7, 'ffn_dim_per_branch': ffn_per_branch,
              'shared_expert_dim': se_dim, 'dropout': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"
fi

echo ""
echo "rtx5090_14 (E-NLP-1) done: $(date)"
