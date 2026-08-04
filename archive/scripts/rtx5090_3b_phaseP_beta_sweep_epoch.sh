#!/bin/bash
###############################################################################
# RTX 5090 Script 3b: NLP mini-ablation — β sweep at Phase P anchor, best pa.
#
# Runs AFTER 3a. Determines best_pa from 3a's 3-seed means (tie-break: larger
# pa wins). If best_pa = 0.5, β sweep is skipped entirely because g = pa − pi
# = 0 there, making gap_c = g · [(1-frac_c)/(1-1/M)]^β = 0 for all c regardless
# of β — per-category routing degenerates to scalar and β becomes inert.
#
# Otherwise: sweeps β ∈ {1.0, 2.0, 4.0} at (best_pa, SE=1.0, wr=1.0 cosine,
# per-cat, domain labels). 3 new seeds × 2 new β values (β=1 reused from
# phase3a_pa${best_pa}_s{seed}) = 6 new runs at most. If Phase V already
# produced some of these cells, they're auto-skipped via results.json.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phase3b_pa{pa}_beta{β}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# Read best_pa from 3a (tie-break: larger pa wins). Env-overridable via BEST_PA.
if [ -z "$BEST_PA" ]; then
    BEST_PA=$($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
rows = {}
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3a_pa{pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if len(ppls) == 3:
        rows[pa] = sum(ppls)/3
if not rows:
    print('', end=''); exit()
best_m = min(rows.values())
tied = [pa for pa, m in rows.items() if m - best_m < 0.05]
print(max(tied, key=float))
EOF
)
fi

if [ -z "$BEST_PA" ]; then
    echo "ERROR: could not read best_pa from phase3a_* results. Run 3a first or set BEST_PA."
    exit 1
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation 3b — β sweep @ Phase P anchor, pa=$BEST_PA"
echo " (uniform + SE=1.0 + per-cat + wr=1.0 cosine, β ∈ {1, 2, 4})"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

# Handle degenerate case: at pa=0.5, g=0 → gap_c=0 ∀c regardless of β.
# β sweep would produce bit-identical results; skip by design.
if [ "$BEST_PA" = "0.5" ]; then
    echo "[3b] best_pa = 0.5 → g = pa − pi = 0 → per-cat ≡ scalar;"
    echo "     β sweep is inert at this operating point. Skipping."
    echo "     Downstream 3c will use β=1 by default."
    exit 0
fi

PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

run_beta() {
    local BETA=$1 SEED=$2
    local ENAME="phase3b_pa${BEST_PA}_beta${BETA}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    # β=1 reuses 3a data at the same pa — symlink-less skip by checking the
    # sibling phase3a_ dir.
    if [ "$BETA" = "1.0" ]; then
        local sibling="${OUTDIR_BASE}/phase3a_pa${BEST_PA}_s${SEED}/results.json"
        if [ -f "$sibling" ]; then
            echo "  $ENAME — reusing phase3a_pa${BEST_PA}_s${SEED} (β=1 identical config)"
            return
        fi
    fi
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BETA $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
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
                   'frac_per_category': fracs, 'amplification_beta': BETA,
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
    echo "[3b β-sweep] seed=$SEED  pa=$BEST_PA"
    for BETA in 1.0 2.0 4.0; do
        run_beta "$BETA" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3b summary (β sweep @ pa=$BEST_PA)"
echo "============================================================"
BEST_PA=$BEST_PA $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA']
rows = {}
for beta in ('1.0', '2.0', '4.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3b_pa{best_pa}_beta{beta}_s{s}', 'results.json')
        # β=1 fallback to phase3a dir
        if not os.path.exists(p) and beta == '1.0':
            p = os.path.join(base, f'phase3a_pa{best_pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows[beta] = (m, std, len(ppls))

print(f"{'β':>5}  {'mean PPL':>10}  {'std':>6}  n")
complete = []
for beta in ('1.0', '2.0', '4.0'):
    if beta in rows:
        m, s, n = rows[beta]
        print(f"{beta:>5}  {m:>10.2f}  {s:>6.3f}  {n}")
        if n == 3:
            complete.append((beta, m))

if complete:
    # Strict argmin on 3-seed mean — matches the most common paper ablation
    # convention (readers see the table, look for the lowest number).
    best_beta, best_m = min(complete, key=lambda x: x[1])
    print(f"\nBEST_β_3b = {best_beta}  (3-seed mean {best_m:.4f} PPL)")
    # Document any near-ties for paper honesty.
    close = [(b, m) for b, m in complete if b != best_beta and m - best_m < 0.05]
    if close:
        others = ', '.join(f'β={b}→{m:.4f}' for b, m in close)
        print(f"  (close within 0.05 PPL of best: {others})")
EOF
echo "Done: $(date)"
