#!/bin/bash
###############################################################################
# RTX 5090 Script 3i: NLP non-uniform branch validation.
#
# Single-cell validation of the data-proportional ffn_dims_per_branch idea.
# Uses CIFAR-optimal hyperparameters (pa=0.7, wr=1.0, SE=0) with per-branch
# FFN width allocated proportional to SlimPajama domain chunk counts in the
# 100M tokenized cache:
#
#   Domain (branch k)       frac       ffn_dim_k
#   --------------------    ------     ---------
#   CommonCrawl    (0)      53.4%      822
#   C4             (1)      18.5%      286
#   Github         (2)       8.8%      136
#   Book           (3)       4.6%       70   ← domain has data but val subset
#                                              may not sample it; branch still exists
#   ArXiv          (4)       6.6%      101
#   Wikipedia      (5)       4.3%       66
#   StackExchange  (6)       3.8%       58
#                                     ------
#                                      1539   (≈ dense ffn 1536, within sanity ±2%)
#
# Rationale (motivated by diagnose_nlp_specialization.py results):
#   - Phase A/F pa=0.7 runs show perfect 6/6 diagonal specialization — branches
#     DO learn domain-specific features, so per-domain capacity matters.
#   - CC/C4 (large, generalist) have high cross-branch leakage (own-branch Δ
#     is only 38% of total branch contribution) → may benefit from more own
#     capacity that can internalize some of the leakage.
#   - ArXiv/Wikipedia (small, specialist) are 82-84% own-branch reliant → may
#     suffer from shrunk capacity. Single-run will reveal net effect.
#
# Compared to uniform ours at pa=0.7 wr=1.0 SE=0 (= Phase F pa=0.7), this is
# a 1-cell A/B: same total ffn budget, just redistributed.
#
# Output: outputs/rtx5090_nlp_mini_ablation/nonuniform_pa0.7_wr1.0_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP non-uniform validation — pa=0.7 wr=1.0 SE=0 (${#SEEDS[@]} seeds)"
echo " Data-proportional ffn_dims: [822, 286, 136, 70, 101, 66, 58]"
echo " $(date)"
echo "============================================================"

run_seed() {
    local SEED=$1
    local ENAME="nonuniform_pa0.7_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    local TMPYAML="${ODIR}/_tmp.yaml"
    # Derive per-branch FFN widths at runtime from the 100M tokenized cache's
    # actual domain_id distribution. This is more robust than hardcoding the
    # list — future cache rebuilds / data changes auto-propagate, and the
    # branch ↔ domain_id ↔ ffn mapping is printed to log for eyeball check.
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import find_tokenize_cache, compute_proportional_ffn_dims

ODIR  = os.environ['ODIR']
ENAME = os.environ['ENAME']
SEED  = int(os.environ['SEED'])

cache = find_tokenize_cache(
    data_dir='./data_cache/slimpajama',
    split='train', max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] using cache: {cache}')

ffn_dims, mapping = compute_proportional_ffn_dims(
    cache_path=cache, total_ffn=1540, num_branches=7)

print(f'[cfg-gen] branch ↔ domain ↔ ffn allocation (domain_id order):')
print(f'  {"branch":<7}{"domain":<30}{"count":>10}{"frac":>9}{"ffn_dim":>10}')
for m in mapping:
    print(f"  b{m['branch']:<6}{m['domain_name']:<30}{m['count']:>10,}{m['frac']:>9.2%}{m['ffn_dim']:>10}")
print(f'[cfg-gen] sum(ffn_dims) = {sum(ffn_dims)} (target 1540, dense FFN 1536)')
print(f'[cfg-gen] largest/smallest = {max(ffn_dims)} / {min(ffn_dims)} = {max(ffn_dims)/max(min(ffn_dims),1):.1f}x')

# Assert branch 0 maps to CommonCrawl (largest domain) and its branch is the
# biggest branch — guards against silent list-ordering regressions.
assert mapping[0]['domain_name'] == 'RedPajamaCommonCrawl', \
    f"branch 0 should be CommonCrawl, got {mapping[0]['domain_name']}"
assert ffn_dims[0] == max(ffn_dims), \
    f"branch 0 (CC, most data) should get the largest ffn, got {ffn_dims[0]} (max {max(ffn_dims)})"

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dims_per_branch': ffn_dims,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {
        'type': 'soft_specdrop', 'p_active': 0.7, 'p_inactive': 0.3,
        'assignment': 'round_robin', 'warmup_ratio': 1.0,
    },
    'training': {
        'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
        'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
        'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead',
    },
    'data': {
        'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
        'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000,
    },
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
print(f'[cfg-gen] wrote {os.path.join(ODIR, "_tmp.yaml")}')
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "$TMPYAML" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[non-uniform validation] seed=$SEED"
    run_seed "$SEED"
done

echo ""
echo "============================================================"
echo " Summary: non-uniform vs uniform (Phase F pa=0.7 reference)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'config':<40} {'PPL':>8} {'std':>6} {'n':>3}")
print('-' * 65)

# Non-uniform (this run)
ppls_nu = []
for s in (42, 123, 456):
    p = os.path.join(base, f'nonuniform_pa0.7_wr1.0_s{s}', 'results.json')
    if os.path.exists(p):
        ppls_nu.append(json.load(open(p))['best_val_ppl'])
if ppls_nu:
    m = sum(ppls_nu) / len(ppls_nu)
    std = (sum((x - m) ** 2 for x in ppls_nu) / len(ppls_nu)) ** 0.5 if len(ppls_nu) > 1 else 0.0
    print(f"{'non-uniform (proportional)':<40} {m:>8.2f} {std:>6.2f} {len(ppls_nu):>3}")

# Phase F pa=0.7 (uniform, same pa/wr/SE)
ppls_u = []
for s in (42, 123, 456):
    p = os.path.join(base, f'phaseF_pa0.7_pi0.3_s{s}', 'results.json')
    if os.path.exists(p):
        ppls_u.append(json.load(open(p))['best_val_ppl'])
if ppls_u:
    m = sum(ppls_u) / len(ppls_u)
    std = (sum((x - m) ** 2 for x in ppls_u) / len(ppls_u)) ** 0.5 if len(ppls_u) > 1 else 0.0
    print(f"{'uniform (Phase F pa=0.7)':<40} {m:>8.2f} {std:>6.2f} {len(ppls_u):>3}")

if ppls_nu and ppls_u:
    delta = (sum(ppls_nu) / len(ppls_nu)) - (sum(ppls_u) / len(ppls_u))
    verdict = ('helps' if delta < -0.1 else 'hurts' if delta > 0.1 else 'neutral')
    print(f"\nΔ (non-uniform − uniform) = {delta:+.2f} PPL  → {verdict}")
    print(f"Phase A pa=0.5 no_routing reference: 56.74 PPL")
    print(f"Phase F pa=0.5 no_routing reference: 56.66 PPL")
EOF
echo "Done: $(date)"
