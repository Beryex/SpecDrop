#!/bin/bash
###############################################################################
# RTX 5090 Script 13: 30M SlimPajama Table 3 — 1 epoch (modern LM training).
#
# Mirror of scripts/experiments/nlp/main_table.sh's structure with ONE change:
#   epochs: 10 → epochs: 1
# Output goes to a SEPARATE directory so the 10-epoch results.json files
# remain untouched.
#
# Why: 30M Table 3 used 10-epoch training (500M unique
# tokens × 10 = 5B token-passes), but modern LM standard is 1-epoch
# (GPT-3 / LLaMA / Mistral all use 1 epoch). The natural objection: "your
# 10-epoch protocol is non-standard; results may not generalize to
# 1-epoch modern training". Counter: re-run the full 8-method × 3-seed
# main-table at 1 epoch and show the ranking is preserved.
#
# Methods (8 = ours_phaseP + 7 baselines, matches paper Table 3):
#   1. Dense TransformerLM        — ffn=1536 (reference, 30.1M)
#   2. Switch Transformer         — N=32, ffn=48
#   3. Hash Layers LM             — N=8, ffn=192
#   4. SMoE-Dropout LM (k=1)      — N=16, ffn=96
#   5. DEMix LM                   — N=7, ffn=220
#   6. MB-LM no-routing (no SE)   — K=7, ffn=220
#   7. Soft SpecDrop (OURS)       — K=7, ffn=205, SE=103, pa=0.6, β=4, step warmup
#   8. MB-LM no-routing + SE=0.5  — K=7, ffn=205, SE=103 (matched scalar to ours)
#
# Per-cell wall: ~30-50 min at bs=64 × 1 epoch on 5090. 8 methods × 3 seeds
# = 24 cells × ~40min avg = ~16h sequential on a single GPU.
#
# Skip-logic: each cell checks its own results.json; safe to re-run.
###############################################################################

set -eo pipefail
PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_30m_1ep"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " SlimPajama × Transformer LM — 1 EPOCH (8 methods × ${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

# Identical to rtx5090_4 except `epochs: 1` (ONLY change).
COMMON_TRAINING="'epochs': 1, 'batch_size': 64, 'max_tokens': 500_000_000, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512"
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

echo "[1/8] Dense TransformerLM (reference)"
run_experiment "dense" "
cfg = {
    'model': {'type': 'transformer_lm', $LM_SHARED, 'ffn_dim': 1536, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[2/8] Switch Transformer (paper-native N=32, per-token top-1 + load-balance)"
run_experiment "switch" "
cfg = {
    'model': {'type': 'switch_transformer_lm', $LM_SHARED, 'num_experts': 32, 'ffn_dim_per_expert': 48, 'load_balance_weight': 0.01, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[3/8] Hash Layers LM (paper-native N=8, per-token hash routing)"
run_experiment "hash_layers" "
cfg = {
    'model': {'type': 'hash_layers_transformer_lm', $LM_SHARED, 'num_experts': 8, 'ffn_dim_per_expert': 192, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[4/8] SMoE-Dropout LM (paper-native N=16, k_init=1, linear k-schedule, no expert-dropout)"
run_experiment "smoe_dropout" "
cfg = {
    'model': {'type': 'smoe_dropout_transformer_lm', $LM_SHARED, 'num_experts': 16, 'ffn_dim_per_expert': 96, 'k_init': 1, 'expert_drop_prob': 0.0, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[5/8] DEMix LM (per-document domain→expert, 7 domains)"
run_experiment "demix" "
cfg = {
    'model': {'type': 'demix_transformer_lm', $LM_SHARED, 'num_domains': 7, 'ffn_dim_per_expert': 220, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING, 'demix_eval_mode': 'mixture'},
    'data': {$COMMON_DATA},
}"

echo "[6/8] MultiBranchTransformerLM no-routing (architectural ref, no SE)"
run_experiment "no_routing" "
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[7/8] Soft SpecDrop (OURS, 100M mini-ablation lock-in: pa=0.6, β=4, SE=0.5 step)"
echo "       frac_per_category auto-loaded from 500M train tokenize cache"
run_experiment "ours_phaseP" "
from data.slimpajama import find_tokenize_cache, compute_category_fractions, split_ffn_budget_for_se
cache = find_tokenize_cache(data_dir='./data_cache/slimpajama', max_tokens=500_000_000, max_seq_len=512)
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[ours_phaseP_1ep] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={7*ffn_per_branch + se_dim}')
print(f'[ours_phaseP_1ep] frac_per_category (from 500M cache): {[round(f, 4) for f in fracs]}')
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

echo "[8/8] MultiBranchTransformerLM no-routing + SE=0.5 (arch-matched scalar to [7])"
run_experiment "no_routing_se05" "
from data.slimpajama import split_ffn_budget_for_se
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[no_routing_se05_1ep] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}')
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED,
              'num_branches': 7, 'ffn_dim_per_branch': ffn_per_branch,
              'shared_expert_dim': se_dim, 'dropout': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo ""
echo "============================================================"
echo " 1-EPOCH Results (SlimPajama 500M tokens × 1 epoch)"
echo "============================================================"
$PYTHON -c "
import json, os
methods = ['dense','switch','hash_layers','smoe_dropout','demix',
            'no_routing','ours_phaseP','no_routing_se05']
labels  = ['Dense','Switch','Hash','SMoE-Drop','DEMix',
            'MB-LM (no SE)','Ours (phaseP, pa=0.6 β=4 SE=0.5 step)',
            'MB-LM+SE=0.5 (arch-matched)']
for m, l in zip(methods, labels):
    ppls = []
    for s in [42, 123, 456]:
        path = f'$OUTDIR_BASE/{m}_s{s}/results.json'
        if not os.path.exists(path): continue
        try:
            d = json.load(open(path))
            if 'best_val_ppl' in d:
                ppls.append(d['best_val_ppl'])
        except (json.JSONDecodeError, OSError):
            pass
    if ppls:
        mean = sum(ppls)/len(ppls)
        std = (sum((x-mean)**2 for x in ppls)/len(ppls))**0.5
        print(f'  {l:42s}: PPL {mean:.2f} +/- {std:.2f}  (n={len(ppls)})')
"
echo "Done: $(date)"
