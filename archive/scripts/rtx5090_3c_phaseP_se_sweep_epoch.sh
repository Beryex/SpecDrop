#!/bin/bash
###############################################################################
# RTX 5090 Script 3c: NLP mini-ablation — SE ratio sweep at Phase P anchor.
#
# Runs AFTER 3a + 3b. Uses best_pa from 3a and best_β from 3b. Sweeps the
# Shared Expert ratio ∈ {0, 0.5, 1.0, 2.0} (scaling SE_dim / avg_branch_width
# while keeping total params ≈ 1540 via split_ffn_budget_for_se).
#
# Reuse: SE=1.0 at (best_pa, best_β=1) is already covered by phase3a_pa{best_pa}_s{s}/
# (since Phase P anchor has SE=1.0 built in). SE=1.0 with best_β ≠ 1 is covered by
# phase3b_pa{best_pa}_beta{best_β}_s{s}/. Script detects these and skips.
#
# New runs (if best_β=1, default): SE ∈ {0, 0.5, 2.0} × 3 seeds = 9 runs.
# If best_β ≠ 1: same 3 SE values × 3 seeds = 9 runs (SE=1.0 from 3b, not 3a).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phase3c_pa{pa}_beta{β}_se{se}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# Read best_pa from 3a
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
if rows:
    best_m = min(rows.values())
    tied = [pa for pa, m in rows.items() if m - best_m < 0.05]
    print(max(tied, key=float))
EOF
)
fi

# Read best_β from 3b (or default to 1.0 if pa=0.5 / 3b skipped)
if [ -z "$BEST_BETA" ]; then
    BEST_BETA=$(BEST_PA=$BEST_PA $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
pa = os.environ['BEST_PA']
if pa == '0.5':
    print('1.0'); exit()
rows = {}
for beta in ('1.0', '2.0', '4.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3b_pa{pa}_beta{beta}_s{s}', 'results.json')
        if not os.path.exists(p) and beta == '1.0':
            p = os.path.join(base, f'phase3a_pa{pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if len(ppls) == 3:
        rows[beta] = sum(ppls)/3
if rows:
    # Strict argmin (matches 3b summary's reporting convention).
    best_beta = min(rows, key=rows.get)
    print(best_beta)
else:
    print('1.0')
EOF
)
fi

if [ -z "$BEST_PA" ] || [ -z "$BEST_BETA" ]; then
    echo "ERROR: could not determine (best_pa, best_β). Run 3a + 3b first, or set env vars."
    exit 1
fi

PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation 3c — SE sweep @ Phase P anchor"
echo " pa=$BEST_PA  β=$BEST_BETA  wr=1.0 cosine  per-cat  (uniform branches)"
echo " SE ∈ {0, 0.5, 1.0, 2.0}, ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_se() {
    local SE=$1 SEED=$2
    local ENAME="phase3c_pa${BEST_PA}_beta${BEST_BETA}_se${SE}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"

    # SE=1.0 reuse. Where does SE=1.0 at (best_pa, best_β) live?
    #   - if best_β = 1.0 → phase3a_pa${BEST_PA}_s${SEED}
    #   - if best_β ≠ 1.0 → phase3b_pa${BEST_PA}_beta${BEST_BETA}_s${SEED}
    if [ "$SE" = "1.0" ]; then
        local sib_a="${OUTDIR_BASE}/phase3a_pa${BEST_PA}_s${SEED}/results.json"
        local sib_b="${OUTDIR_BASE}/phase3b_pa${BEST_PA}_beta${BEST_BETA}_s${SEED}/results.json"
        if [ "$BEST_BETA" = "1.0" ] && [ -f "$sib_a" ]; then
            echo "  $ENAME — reusing phase3a_pa${BEST_PA}_s${SEED} (SE=1.0 β=1 identical config)"
            return
        fi
        if [ -f "$sib_b" ]; then
            echo "  $ENAME — reusing phase3b_pa${BEST_PA}_beta${BEST_BETA}_s${SEED}"
            return
        fi
    fi

    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BEST_BETA SE_RATIO=$SE \
      $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA']); SE_RATIO = float(os.environ['SE_RATIO'])

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
    echo "[3c SE-sweep] seed=$SEED  pa=$BEST_PA  β=$BEST_BETA"
    for SE in 0 0.5 1.0 2.0; do
        run_se "$SE" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3c summary (SE sweep @ pa=$BEST_PA β=$BEST_BETA)"
echo "============================================================"
BEST_PA=$BEST_PA BEST_BETA=$BEST_BETA $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
pa = os.environ['BEST_PA']; beta = os.environ['BEST_BETA']
rows = {}
for se in ('0', '0.5', '1.0', '2.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3c_pa{pa}_beta{beta}_se{se}_s{s}', 'results.json')
        # SE=1.0 fallback to 3a (if β=1) or 3b
        if not os.path.exists(p) and se == '1.0':
            if beta == '1.0':
                p = os.path.join(base, f'phase3a_pa{pa}_s{s}', 'results.json')
            else:
                p = os.path.join(base, f'phase3b_pa{pa}_beta{beta}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows[se] = (m, std, len(ppls))

print(f"{'SE':>5}  {'mean PPL':>10}  {'std':>6}  n")
complete = []
for se in ('0', '0.5', '1.0', '2.0'):
    if se in rows:
        m, s, n = rows[se]
        print(f"{se:>5}  {m:>10.2f}  {s:>6.3f}  {n}")
        if n == 3:
            complete.append((se, m))

if complete:
    # Strict argmin (matches 3b's reporting convention; readers see the table minimum).
    best_se, best_m = min(complete, key=lambda x: x[1])
    print(f"\nBEST_SE_3c = {best_se}  (3-seed mean {best_m:.4f} PPL)")
    close = [(s, m) for s, m in complete if s != best_se and m - best_m < 0.05]
    if close:
        others = ', '.join(f'SE={s}→{m:.4f}' for s, m in close)
        print(f"  (close within 0.05 PPL of best: {others})")
    print(f"\nFinal Phase P+ablation config: pa={pa}  β={beta}  SE_ratio={best_se}  wr=1.0 cosine")
EOF
echo "Done: $(date)"
