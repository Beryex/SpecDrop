#!/bin/bash
###############################################################################
# RTX 5090 Script 3b: NLP mini-ablation — β sweep at (best_pa, anchor_SE).
#
# METHOD UPDATE (2026-04-21): Step warmup + strict argmin excluding pa=0.5
# as the structural degenerate point. At pa=pi=0.5, g = p_a − p_i = 0, so
# gap_c = g·ratio = 0 for all categories and all β; the per-category
# mechanism is mathematically inactive regardless of β. We keep pa=0.5
# runs as the β-invariant reference baseline but drop it from the "best
# method config" search (select_best_pa(exclude_pa=['0.5'])).
#
# Chain logic (runs AFTER 3a@SE=0):
#   1. Barrier on 3a SE=0 (18 results.json).
#   2. literal argmin (no exclusion) at SE=0.
#   3. IF literal argmin == '0.5' (degenerate wins overall at SE=0):
#        → SE=0 is not the method regime (scalar beats per-cat at SE=0).
#          Fall back to SE=1.0 where shared expert gives per-cat headroom.
#        a. Run 3a at ANCHOR_SE=1.0 for this GPU's seed.
#        b. Barrier on all 3 GPUs' 3a@SE=1.0.
#        c. best_pa = strict argmin @ SE=1.0 excluding pa=0.5.
#        d. ANCHOR_SE = '1.0'.
#      ELSE:
#        best_pa = strict argmin @ SE=0 excluding pa=0.5.
#        ANCHOR_SE = '0'.
#   4. Write "$OUTDIR_BASE/_anchor_se.txt" + "_best_pa.txt" (3c reads).
#   5. β sweep at (best_pa, ANCHOR_SE). β=1.0 reuses 3a cell (free).
#
# ENV OVERRIDE: Set BEST_PA and ANCHOR_SE before running to skip auto-detect
# entirely (used when the user has resolved those offline; see _run_gpu*.sh).
#
# Sweep: β ∈ {0, 1.0, 2.0, 4.0}. Net new runs: 3 β × 3 seeds = 9.
# (β=1.0 reuses 3a cell at best_pa, free reference.)
#
# β=0 is the scalar-per-cat reference: at β=0, ratio_c = ((1-frac_c)/(1-1/M))^0 = 1
# for all c, so gap_c = g (constant across categories). The per-category routing
# machinery is active (p_active_c, p_inactive_c differ from pa=0.5 uniform), but
# gets NO per-category differentiation. This isolates "pa > 0.5" (mechanism ON)
# from "β > 0" (category differentiation). Paper story:
#     pa=0.5:      mechanism OFF      (g=0)
#     pa=0.6 β=0:  mechanism ON, no differentiation  (g>0, uniform ratio)
#     pa=0.6 β>0:  mechanism ON, per-cat differentiation
#
# Selection: strict argmin (exact tie → smaller β).
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on 3a@SE=0, auto-detect best_pa + anchor_SE ───────────────────
if [ -z "$ANCHOR_SE" ] && [ -z "$BEST_PA" ]; then
    echo "[3b] Waiting for 3a@SE=0 to complete (18 results.json files)..."
    $PYTHON -m scripts.ablation_chain wait --phase 3a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se 0 --poll 60

    # Literal argmin (no exclusion) — if pa=0.5 wins, SE=0 is not the regime.
    LITERAL_SE0=$($PYTHON -m scripts.ablation_chain best --phase 3a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 --anchor-se 0)
    if [ -z "$LITERAL_SE0" ]; then
        echo "ERROR: could not determine literal argmin from 3a@SE=0"; exit 1
    fi
    echo "[3b] 3a@SE=0 literal argmin → pa = $LITERAL_SE0"

    if [ "$LITERAL_SE0" = "0.5" ]; then
        echo ""
        echo "=========================================================="
        echo "[3b] DEGENERATE AT SE=0 — 3a@SE=0 literal argmin is pa=0.5"
        echo "     (at pa=pi=0.5, g=0 → gap_c=0 ∀c,β → per-cat mechanism"
        echo "     inactive). SE=0 is not the method regime. Falling back"
        echo "     to SE=1.0 (shared-expert anchor for the 500M ours)."
        echo "=========================================================="
        ANCHOR_SE=1.0 bash scripts/experiments/nlp/ablation_pa.sh

        echo "[3b] Waiting for 3a@SE=1.0 to complete on all 3 GPUs..."
        $PYTHON -m scripts.ablation_chain wait --phase 3a --base "$OUTDIR_BASE" \
            --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
            --anchor-se 1.0 --poll 60

        # Exclude pa=0.5 (degenerate) from the mechanism-active argmin.
        BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 3a --base "$OUTDIR_BASE" \
            --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
            --anchor-se 1.0 --exclude-pa 0.5)
        if [ -z "$BEST_PA" ]; then
            echo "ERROR: could not determine BEST_PA from 3a@SE=1.0"; exit 1
        fi
        ANCHOR_SE="1.0"
        echo "[3b] 3a@SE=1.0 mechanism argmin (excluding pa=0.5) → best_pa = $BEST_PA"
    else
        # SE=0 is the method regime. Still exclude pa=0.5 defensively.
        BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 3a --base "$OUTDIR_BASE" \
            --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
            --anchor-se 0 --exclude-pa 0.5)
        if [ -z "$BEST_PA" ]; then
            echo "ERROR: could not determine BEST_PA from 3a@SE=0"; exit 1
        fi
        ANCHOR_SE="0"
        echo "[3b] 3a@SE=0 mechanism argmin (excluding pa=0.5) → best_pa = $BEST_PA"
    fi
fi

# Env-overridable (for manual re-runs / debugging)
ANCHOR_SE=${ANCHOR_SE:-0}
if [ -z "$BEST_PA" ]; then echo "ERROR: BEST_PA not set"; exit 1; fi
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

# Persist anchor decision for 3c (idempotent)
echo "$ANCHOR_SE" > "$OUTDIR_BASE/_anchor_se.txt"
echo "$BEST_PA" > "$OUTDIR_BASE/_best_pa.txt"

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

echo ""
echo "============================================================"
echo " NLP mini-ablation 3b — β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
echo " (uniform + per-cat + wr=1.0 cosine, warmup_unit=step)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_beta() {
    local BETA=$1 SEED=$2
    local ENAME="phase3b_pa${BEST_PA}_beta${BETA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    # β=1.0 at (best_pa, ANCHOR_SE) IS the corresponding 3a cell; reuse.
    if [ "$BETA" = "1.0" ]; then
        local sibling="${OUTDIR_BASE}/phase3a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}/results.json"
        if [ -f "$sibling" ]; then
            echo "  $ENAME — reusing phase3a_pa${BEST_PA}${SE_SUFFIX}_s${SEED} (β=1 identical)"
            return
        fi
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BETA ANCHOR_SE=$ANCHOR_SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

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
    echo "[3b β-sweep] seed=$SEED  pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
    for BETA in 0 1.0 2.0 4.0; do
        run_beta "$BETA" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 3b seed-local summary (β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE)"
echo "============================================================"
BEST_PA=$BEST_PA ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA']; anchor = os.environ['ANCHOR_SE']; suf = os.environ['SE_SUFFIX']
print(f"All runs: pa={best_pa} SE={anchor} per-cat step-warmup uniform.")
print(f"β=1.0 from 3a reference; β∈{{0,2,4}} are new runs.")
print(f"β=0: scalar per-cat reference (g>0 but no differentiation).\n")
print(f"{'β':>4}  {'mean PPL':>10}  {'std':>6}  n")
for beta in ('0', '1.0', '2.0', '4.0'):
    ppls = []
    for s in (42, 123, 456):
        if beta == '1.0':
            p = os.path.join(base, f'phase3a_pa{best_pa}{suf}_s{s}', 'results.json')
        else:
            p = os.path.join(base, f'phase3b_pa{best_pa}_beta{beta}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        sd = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        print(f"{beta:>4}  {m:>10.2f}  {sd:>6.3f}  {len(ppls)}")
print(f"\n[3b ANCHOR_SE={anchor}] strict argmin will be computed by 3c.")
EOF
echo "Done: $(date)"
