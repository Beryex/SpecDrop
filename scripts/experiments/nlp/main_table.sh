#!/bin/bash
###############################################################################
# RTX 5090 Script 4 (FAITHFUL): SlimPajama × Transformer LM, native baselines.
#
# Each baseline uses its paper-native N, with per-expert FFN width narrowed so
# N × ffn_per_expert ≈ 1536 (dense MLP hidden) — keeps total params ≈ 30M across
# the comparison.
#
#  1. Dense TransformerLM                     — ffn=1536 (reference, 30.1M)
#  2. Switch Transformer    (Fedus 2022)      — N=32, ffn=48   (param-matched top-1)
#  3. Hash Layers           (Roller 2021)     — N=8,  ffn=192  (paper small-GPT)
#  4. SMoE-Dropout          (Chen 2023)       — N=16, ffn=96   (paper BERT-base)
#  5. DEMix                 (Gururangan 2022) — N=7,  ffn=220  (= #domains)
#  6. MB-TransformerLM no-routing             — K=7, ffn=220  (arch ref, no SE)
#  7. Soft SpecDrop (ours legacy)             — K=7, ffn=220  pa=0.7 wr=1.0 no SE
#                                                (initial configuration — kept
#                                                 as provenance; superseded by [9])
#  8. MB-TransformerLM no-routing + SE=1.0    — K=7, ffn=192, SE=192
#                                                (paired with [9] ours variants)
#  9. Soft SpecDrop (OURS) — 100M mini-ablation lock-in (3a+3b+3c step)
#                              — K=7 uniform, ffn=205, SE=103 (SE_ratio=0.5)
#                                pa=0.6, pi=0.4, per-cat β=4,
#                                wr=1.0 cosine, warmup_unit='step',
#                                per-category fracs from 500M train cache
# 10. MB-TransformerLM no-routing + SE=0.5    — K=7, ffn=205, SE=103
#                                                (NOTE: mathematically ≡ pa=0.5
#                                                 SoftSpecDrop at SE=0.5 —
#                                                 at pa=pi=0.5 the mask matrix
#                                                 is constant 0.5, giving
#                                                 per-branch contribution
#                                                 0.5/S = 0.5/3.5 = 1/7, which
#                                                 is identical to uniform 1/K.
#                                                 So [10] serves BOTH as the
#                                                 architecture-matched scalar
#                                                 reference AND as the method's
#                                                 mechanism-OFF (β-invariant)
#                                                 degenerate point at matched
#                                                 SE. Critical baseline for
#                                                 [9].)
#
# Final paper ablation outcome (rtx5090_3a → 3b → 3c step warmup, 100M tokens):
#   pa sweep at (SE=1.0, β=1, step): literal argmin pa=0.5 (55.083, degenerate);
#   excluding pa=0.5, argmin is pa=0.6 (55.137, mechanism argmin). SE=0 literal
#   argmin also pa=0.5 → confirmed degenerate regime at SE=0; anchor shifts to
#   SE=1.0 where shared expert gives per-cat amplification room.
#   β sweep at (pa=0.6, SE=1.0): β=4 strict argmin (55.130 vs β=1 55.137 —
#   all 4 β values within 0.04 PPL, β statistically tied).
#   SE sweep at (pa=0.6, β=4): SE=0.5 strict argmin (55.010 vs SE=1.0 55.130).
#   Method vs scalar (pa=0.5 SE=1.0=55.083): Δ = −0.07 PPL, within 0.6σ noise.
#   Honest methodology: method ties the scalar baseline at 100M SlimPajama.
#
# Param-match (SE_ratio=0.5): split_ffn_budget_for_se(1540, 0.5, 7)
#   → avg_branch = 1540/7.5 ≈ 205, SE = 1540×0.5/7.5 ≈ 103
#   → total = 7×205 + 103 = 1538 (within 0.1% of dense 1536, sanity-check passes).
#
# Setting:
#   Optimizer: AdamW lr=3e-4, wd=0.1, β=(0.9,0.95)
#   Schedule:  Linear warmup (1000 steps) + cosine decay
#   Tokens:    500M, Batch: 64, max_seq_len=512, Seeds: 42 / 123 / 456
###############################################################################

# Fail loudly on any command error — previously a failing config-gen block
# (e.g., find_tokenize_cache FileNotFoundError) let the script continue into
# run_nlp.py with a stale _tmp_s{seed}.yaml from the PREVIOUS cell, silently
# re-running the wrong experiment under the current cell's dir name.
set -eo pipefail
PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_faithful"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " SlimPajama × Transformer LM FAITHFUL (10 × ${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

# NLP paper-canonical effective batch_size is 32. Pre-2026-04-24, run_nlp.py
# silently fell back to data.batch_size default 32, so YAMLs that wrote
# training.batch_size:64 actually trained at 32. All Tab 3 numbers are from
# bs=32. After the run_nlp.py loader fix, the loader correctly
# respects training.batch_size — to reproduce paper-canonical we write 32.
COMMON_TRAINING="'epochs': 10, 'batch_size': 32, 'max_tokens': 500_000_000, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
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
        # Use a per-cell yaml filename so a stale previous-cell yaml can never
        # silently be consumed if config-gen fails here.
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

echo "[1/10] Dense TransformerLM (reference)"
run_experiment "dense" "
cfg = {
    'model': {'type': 'transformer_lm', $LM_SHARED, 'ffn_dim': 1536, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[2/10] Switch Transformer (N=32, per-token top-1 + load-balance)"
run_experiment "switch" "
cfg = {
    'model': {'type': 'switch_transformer_lm', $LM_SHARED, 'num_experts': 32, 'ffn_dim_per_expert': 48, 'load_balance_weight': 0.01, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[3/10] Hash Layers LM (paper-native N=8, per-token hash routing)"
run_experiment "hash_layers" "
cfg = {
    'model': {'type': 'hash_layers_transformer_lm', $LM_SHARED, 'num_experts': 8, 'ffn_dim_per_expert': 192, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[4/10] SMoE-Dropout LM (paper-native N=16, fixed random + linear k-schedule 1→N, no expert-dropout)"
run_experiment "smoe_dropout" "
cfg = {
    'model': {'type': 'smoe_dropout_transformer_lm', $LM_SHARED, 'num_experts': 16, 'ffn_dim_per_expert': 96, 'k_init': 1, 'expert_drop_prob': 0.0, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[5/10] DEMix LM (per-document domain→expert, 7 domains; faithful MoE-posterior eval)"
run_experiment "demix" "
cfg = {
    'model': {'type': 'demix_transformer_lm', $LM_SHARED, 'num_domains': 7, 'ffn_dim_per_expert': 220, 'dropout': 0.1},
    'algorithm': {'type': 'none'},
    'training': {$COMMON_TRAINING, 'demix_eval_mode': 'mixture'},
    'data': {$COMMON_DATA},
}"

echo "[6/10] MultiBranchTransformerLM no-routing (architectural ref)"
run_experiment "no_routing" "
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

echo "[7/10] Soft SpecDrop (ours legacy — superseded by [9])"
echo "       pa=0.7, wr=1.0, no SE, uniform ffn=220 — initial configuration (kept for provenance)."
echo "       Preserved for provenance; replaced by [9] as the paper-ready ours row."
run_experiment "ours" "
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1},
    'algorithm': {'type': 'soft_specdrop', 'p_active': 0.7, 'p_inactive': 0.3, 'assignment': 'round_robin', 'warmup_ratio': 1.0},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

# Param-match (SE ratio=1.0): ffn=192, SE=192 → 7×192 + 192 = 1536 (= dense ffn).
echo "[8/10] MultiBranchTransformerLM no-routing + SE=1.0 (legacy arch-matched baseline)"
echo "        Originally paired with ours+SE=1.0; kept for full SE-curve context."
run_experiment "no_routing_se" "
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 192, 'shared_expert_dim': 192, 'dropout': 0.1},
    'algorithm': {'type': 'no_dropout'},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}"

# Final ours: Phase P-lock-in config from 3a/3b/3c ablation on 100M tokens.
# Per-category fractions are computed at script-gen time from the 500M
# tokenize cache (different from 100M), so ours on 500M uses the actual
# 500M domain distribution — not the 100M proportions that the ablation
# saw. SE_ratio=0.5 → ffn_per_branch=205, SE_dim=103 via split_ffn_budget_for_se.
echo "[9/10] Soft SpecDrop (OURS, 100M mini-ablation lock-in: 3a+3b+3c-step)"
echo "        pa=0.6, pi=0.4, per-cat β=4, SE_ratio=0.5 (ffn=205, SE=103),"
echo "        wr=1.0 cosine, warmup_unit=step"
echo "        frac_per_category auto-loaded from 500M train tokenize cache"
run_experiment "ours_phaseP" "
from data.slimpajama import find_tokenize_cache, compute_category_fractions, split_ffn_budget_for_se
cache = find_tokenize_cache(data_dir='./data_cache/slimpajama', max_tokens=500_000_000, max_seq_len=512)
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[ours_phaseP] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={7*ffn_per_branch + se_dim}')
print(f'[ours_phaseP] frac_per_category (from 500M cache): {[round(f, 4) for f in fracs]}')
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

# Arch-matched baseline for [9]: same K, ffn, SE as ours_phaseP but no routing
# (uniform 1/K per-branch weights). Isolates per-cat routing contribution.
echo "[10/10] MultiBranchTransformerLM no-routing + SE=0.5 (arch-matched to [9])"
echo "        K=7, ffn=205, SE=103 — same architectural capacity as ours_phaseP"
run_experiment "no_routing_se05" "
from data.slimpajama import split_ffn_budget_for_se
total_branch_ffn, se_dim = split_ffn_budget_for_se(total_ffn_budget=1540, se_ratio=0.5, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[no_routing_se05] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}')
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
echo " NLP FAITHFUL Results (SlimPajama 500M tokens)"
echo "============================================================"
$PYTHON -c "
import json, os
methods = ['dense','switch','hash_layers','smoe_dropout','demix',
            'no_routing','ours','no_routing_se','ours_phaseP','no_routing_se05']
labels  = ['Dense','Switch','Hash','SMoE-Drop','DEMix',
            'MB-LM','Ours (legacy pa=0.7)','MB-LM+SE=1.0',
            'Ours (Phase P pa=0.6 β=4 SE=0.5)','MB-LM+SE=0.5 (arch-matched)']
for m, l in zip(methods, labels):
    ppls = []
    for s in [42, 123, 456]:
        path = f'$OUTDIR_BASE/{m}_s{s}/results.json'
        if os.path.exists(path):
            ppls.append(json.load(open(path))['best_val_ppl'])
    if ppls:
        mean = sum(ppls)/len(ppls)
        std = (sum((x-mean)**2 for x in ppls)/len(ppls))**0.5
        print(f'  {l:38s}: PPL {mean:.2f} +/- {std:.2f}  (n={len(ppls)})')
"
echo "Done: $(date)"
