#!/bin/bash
###############################################################################
# RTX 5090 Script 3c: NLP mini-ablation — SE ratio sweep at (best_pa, best_β).
#
# METHOD UPDATE (2026-04-20): Step warmup + strict argmin + SE=0/1.0 anchor
# protocol (see 3a/3b headers). Archived epoch variant:
# archive/scripts/rtx5090_3c_phaseP_se_sweep_epoch.sh.
#
# Runs AFTER 3a AND 3b. Reads the anchor SE chosen by 3b (via
# "$OUTDIR_BASE/_anchor_se.txt"). Barriers on 3a + 3b at that anchor, then
# picks best_β via `scripts.ablation_chain best`.
#
# Sweep: SE_ratio ∈ {0, 0.5, 1.0, 2.0} at (best_pa, best_β). SE at anchor
# (either 0 or 1.0) reuses the corresponding 3a/3b cell (free reference).
# Net new runs: 3 SE × 3 seeds = 9.
#
# Selection: strict argmin (exact tie → smaller SE).
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Read anchor decision from 3b ──────────────────────────────────────────
if [ -z "$ANCHOR_SE" ]; then
    if [ -f "$OUTDIR_BASE/_anchor_se.txt" ]; then
        ANCHOR_SE=$(cat "$OUTDIR_BASE/_anchor_se.txt" | tr -d '[:space:]')
        echo "[3c] Read ANCHOR_SE=$ANCHOR_SE from _anchor_se.txt (set by 3b)"
    else
        # 3b hasn't written marker yet. Infer: if 3a@SE=0 argmin != 0.5, anchor=0.
        echo "[3c] _anchor_se.txt missing; deriving from 3a@SE=0."
        $PYTHON -m scripts.ablation_chain wait --phase 3a --base "$OUTDIR_BASE" \
            --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
            --anchor-se 0 --poll 60
        BEST_PA_SE0=$($PYTHON -m scripts.ablation_chain best --phase 3a --base "$OUTDIR_BASE" \
            --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 --anchor-se 0)
        if [ "$BEST_PA_SE0" = "0.5" ]; then
            ANCHOR_SE="1.0"
        else
            ANCHOR_SE="0"
        fi
    fi
fi

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

# ── Barrier on 3a at the chosen anchor (need best_pa) ─────────────────────
# Prefer reading from 3b's marker file (which encodes the exclude_pa choice).
if [ -z "$BEST_PA" ] && [ -f "$OUTDIR_BASE/_best_pa.txt" ]; then
    BEST_PA=$(cat "$OUTDIR_BASE/_best_pa.txt" | tr -d '[:space:]')
    echo "[3c] Read BEST_PA=$BEST_PA from _best_pa.txt (set by 3b)"
fi
if [ -z "$BEST_PA" ]; then
    echo "[3c] Waiting for 3a@SE=$ANCHOR_SE to complete..."
    $PYTHON -m scripts.ablation_chain wait --phase 3a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --poll 60
    # Exclude pa=0.5 (degenerate point) from the mechanism search.
    BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 3a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --exclude-pa 0.5)
    if [ -z "$BEST_PA" ]; then
        echo "ERROR: could not determine BEST_PA from 3a@SE=$ANCHOR_SE"; exit 1
    fi
fi
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

# ── Barrier on 3b at the chosen anchor (need best_β) ──────────────────────
if [ -z "$BEST_BETA" ]; then
    echo "[3c] Waiting for 3b@SE=$ANCHOR_SE to complete (β sweep at pa=$BEST_PA)..."
    $PYTHON -m scripts.ablation_chain wait --phase 3b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE" --poll 60
    BEST_BETA=$($PYTHON -m scripts.ablation_chain best --phase 3b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE")
    if [ -z "$BEST_BETA" ]; then
        echo "ERROR: could not determine BEST_BETA from 3b@SE=$ANCHOR_SE"; exit 1
    fi
fi

echo ""
echo "============================================================"
echo " NLP mini-ablation 3c — SE sweep @ pa=$BEST_PA  β=$BEST_BETA"
echo " ANCHOR_SE=$ANCHOR_SE (anchor's SE reuses 3a/3b cell)"
echo " (uniform + per-cat + wr=1.0 cosine, warmup_unit=step)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_se() {
    local SE=$1 SEED=$2
    local ENAME="phase3c_pa${BEST_PA}_beta${BEST_BETA}_se${SE}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    # Reuse anchor cell: SE == ANCHOR_SE → read from 3a (if β=1) or 3b (else).
    if [ "$SE" = "$ANCHOR_SE" ]; then
        if [ "$BEST_BETA" = "1.0" ]; then
            local sib="${OUTDIR_BASE}/phase3a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}/results.json"
        else
            local sib="${OUTDIR_BASE}/phase3b_pa${BEST_PA}_beta${BEST_BETA}${SE_SUFFIX}_s${SEED}/results.json"
        fi
        if [ -f "$sib" ]; then
            echo "  $ENAME — reusing $(basename $(dirname $sib))"
            return
        fi
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BEST_BETA SE=$SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['SE'])

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
fracs = compute_category_fractions(cache, num_categories=7)
assert abs(sum(fracs) - 1.0) < 1e-4

print(f'[cfg-gen] pa={PA} β={BETA} SE_ratio={SE_RATIO} warmup_unit=step')
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
                   'frac_per_category': fracs, 'amplification_beta': BETA,
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
    echo "[3c SE-sweep] seed=$SEED  pa=$BEST_PA β=$BEST_BETA ANCHOR_SE=$ANCHOR_SE"
    for SE in 0 0.5 1.0 2.0; do
        run_se "$SE" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3c seed-local summary (SE sweep @ pa=$BEST_PA β=$BEST_BETA)"
echo "============================================================"
BEST_PA=$BEST_PA BEST_BETA=$BEST_BETA ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA']; best_beta = os.environ['BEST_BETA']
anchor = os.environ['ANCHOR_SE']; suf = os.environ['SE_SUFFIX']
print(f"All runs: pa={best_pa} β={best_beta} per-cat step-warmup uniform, ANCHOR_SE={anchor}.\n")
print(f"{'SE':>4}  {'mean PPL':>10}  {'std':>6}  n")
for se in ('0', '0.5', '1.0', '2.0'):
    ppls = []
    for s in (42, 123, 456):
        if se == anchor:
            if best_beta == '1.0':
                p = os.path.join(base, f'phase3a_pa{best_pa}{suf}_s{s}', 'results.json')
            else:
                p = os.path.join(base, f'phase3b_pa{best_pa}_beta{best_beta}{suf}_s{s}', 'results.json')
        else:
            p = os.path.join(base, f'phase3c_pa{best_pa}_beta{best_beta}_se{se}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        sd = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        print(f"{se:>4}  {m:>10.2f}  {sd:>6.3f}  {len(ppls)}")

# Strict argmin
import subprocess, sys
try:
    proc = subprocess.run([sys.executable, '-m', 'scripts.ablation_chain', 'best',
                           '--phase', '3c', '--base', base,
                           '--best-pa', best_pa, '--best-beta', best_beta,
                           '--se-values', '0,0.5,1.0,2.0',
                           '--anchor-se', anchor],
                          capture_output=True, text=True, timeout=10)
    if proc.returncode == 0:
        print(f"\n[3c strict argmin] best_SE = {proc.stdout.strip()}")
except Exception as e:
    print(f"\n(best_SE helper not available: {e})")
EOF
echo ""
echo "Final ours config (step warmup, ANCHOR_SE=$ANCHOR_SE):"
echo "  pa=$BEST_PA  β=$BEST_BETA  warmup_unit=step  anchor_SE=$ANCHOR_SE"
echo "  SE_ratio = (see 3c strict argmin above)"
echo ""
echo "Done: $(date)"
