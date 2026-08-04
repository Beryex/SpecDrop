#!/bin/bash
###############################################################################
# RTX 5090 Script 3a: NLP mini-ablation — pa sweep at Phase P anchor.
#
# Anchor (fixed across 3a/3b/3c): uniform branches + SE_ratio=1.0 +
# per-category routing (β=1) + wr=1.0 cosine warmup + domain labels.
#
# Sweep: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa, 3 seeds each = 18 runs.
# Tie-break (end-of-script summary): if multiple pa values achieve the same
# 3-seed mean PPL within noise, the LARGER pa wins (convention).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phase3a_pa{pa}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation 3a — pa sweep @ Phase P anchor"
echo " (uniform + SE=1.0 + per-cat β=1 + wr=1.0 cosine)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phase3a_pa${PA}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, DOMAIN_NAMES)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': ffn_per_branch,
        'shared_expert_dim': se_dim,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': 1.0,
                   'warmup_schedule': 'cosine'},
    'training': {'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
                  'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
                  'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'},
    'data': {'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
              'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[3a pa-sweep] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3a summary (pa sweep @ Phase P anchor)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
rows = {}
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3a_pa{pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows[pa] = (m, std, len(ppls))

print(f"{'pa':>4}  {'mean PPL':>10}  {'std':>6}  n")
complete = []
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    if pa in rows:
        m, s, n = rows[pa]
        print(f"{pa:>4}  {m:>10.2f}  {s:>6.3f}  {n}")
        if n == 3:
            complete.append((pa, m))

if complete:
    # Tie-break rule: pick LARGER pa among ties (within noise ~ 0.05 PPL).
    # "Tie" = within 0.05 PPL of the absolute minimum.
    best_m = min(m for _, m in complete)
    tied = [pa for pa, m in complete if m - best_m < 0.05]
    best_pa = max(tied, key=float)  # larger pa wins among tied
    print(f"\nBEST_PA_3a = {best_pa}  (3-seed mean {rows[best_pa][0]:.2f} PPL)")
    if len(tied) > 1:
        print(f"  tied within 0.05 PPL: {tied}; tie-break → larger pa = {best_pa}")
EOF
echo "Done: $(date)"
