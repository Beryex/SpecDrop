#!/bin/bash
###############################################################################
# RTX 5090 Script 3a: NLP mini-ablation — pa sweep at Phase P anchor.
#
# METHOD UPDATE (2026-04-20): Step warmup + strict-argmin selection, with
# SE=0 primary anchor (CIFAR-matched protocol) and SE=1.0 fallback when the
# 3a primary search returns pa=0.5 (the degenerate case: g=0 → per-cat gate
# inactive). Archived epoch variant at archive/scripts/*_epoch.sh.
#
# This single script handles both the primary (SE=0) and fallback (SE=1.0)
# passes via the ANCHOR_SE env var:
#   ANCHOR_SE=0   (default)  → outputs phase3a_pa{pa}_s{seed}/,        SE=0
#   ANCHOR_SE=1.0            → outputs phase3a_pa{pa}_se1.0_s{seed}/,  SE=1
#
# Anchor config (fixed across 3a/3b/3c): uniform branches + per-cat routing
# (β=1 in 3a) + wr=1.0 cosine warmup + warmup_unit=step + domain labels.
#
# Sweep: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa, 3 seeds each = 18 runs.
# Selection: strict argmin on 3-seed mean PPL (exact tie → smaller wins).
# No tie-break window.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
ANCHOR_SE=${ANCHOR_SE:-0}
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi
# PA_VALUES env: comma-separated subset of pa values to run (e.g. "0.5" for a
# single-cell matched-SE baseline). Default: full sweep.
if [ -n "$PA_VALUES" ]; then
    PA_LIST=($(echo "$PA_VALUES" | tr ',' ' '))
else
    PA_LIST=(0.5 0.6 0.7 0.8 0.9 1.0)
fi

# Dir-name suffix: empty when ANCHOR_SE=0 (default), '_se{ANCHOR_SE}' otherwise.
if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation 3a — pa sweep @ Phase P anchor  (ANCHOR_SE=${ANCHOR_SE})"
echo " (uniform + per-cat β=1 + wr=1.0 cosine, warmup_unit=step)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phase3a_pa${PA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI ANCHOR_SE=$ANCHOR_SE $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

print(f'[cfg-gen] pa={PA} β=1.0 SE_ratio={SE_RATIO} warmup_unit=step')
print(f'[cfg-gen] ffn_per_branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={7*ffn_per_branch + se_dim}')

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
                   'warmup_schedule': 'cosine',
                   'warmup_unit': 'step'},
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
    echo "[3a pa-sweep ANCHOR_SE=$ANCHOR_SE PA_LIST=${PA_LIST[*]}] seed=$SEED"
    for PA in "${PA_LIST[@]}"; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3a seed-local summary (this GPU's seed completed, ANCHOR_SE=$ANCHOR_SE)"
echo "============================================================"
ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
suf = os.environ['SE_SUFFIX']
anchor = os.environ['ANCHOR_SE']
print(f"{'pa':>4}  {'mean PPL':>10}  {'std':>6}  n")
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase3a_pa{pa}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        sd = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        print(f"{pa:>4}  {m:>10.2f}  {sd:>6.3f}  {len(ppls)}")
print(f"\n[3a ANCHOR_SE={anchor}] strict argmin will be computed by 3b/3c "
      "via scripts.ablation_chain once all 3 seeds' runs land.")
EOF
echo "Done: $(date)"
