#!/bin/bash
###############################################################################
# RTX 5090 Script 5c: ImageNet ViT mini-ablation — SE sweep at (best_pa, best_β).
#
# Mirrors NLP 3c for ViT. Runs AFTER 5a AND 5b. Reads the anchor SE fixed by
# 5b (via "$OUTDIR_BASE/_anchor_se.txt", default 1.0). Barriers on 5a + 5b at
# that anchor, then picks best_β via `scripts.ablation_chain best --maximize
# --metric-key best_top1`.
#
# Sweep: SE_ratio ∈ {0, 0.5, 1.0, 2.0} at (best_pa, best_β). SE at anchor
# (1.0) reuses the corresponding 5a/5b cell (free reference).
# Net new runs: 3 SE × 3 seeds = 9.
#
# Selection: strict argmax on top1 (exact tie → smaller SE).
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_vit_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Read anchor decision from 5b ──────────────────────────────────────────
if [ -z "$ANCHOR_SE" ]; then
    if [ -f "$OUTDIR_BASE/_anchor_se.txt" ]; then
        ANCHOR_SE=$(cat "$OUTDIR_BASE/_anchor_se.txt" | tr -d '[:space:]')
        echo "[5c] Read ANCHOR_SE=$ANCHOR_SE from _anchor_se.txt (set by 5b)"
    else
        echo "[5c] _anchor_se.txt missing; defaulting to ANCHOR_SE=1.0 (5a/5b anchor)."
        ANCHOR_SE="1.0"
    fi
fi

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

# ── Barrier on 5a at the chosen anchor (need best_pa) ─────────────────────
if [ -z "$BEST_PA" ] && [ -f "$OUTDIR_BASE/_best_pa.txt" ]; then
    BEST_PA=$(cat "$OUTDIR_BASE/_best_pa.txt" | tr -d '[:space:]')
    echo "[5c] Read BEST_PA=$BEST_PA from _best_pa.txt (set by 5b)"
fi
if [ -z "$BEST_PA" ]; then
    echo "[5c] Waiting for 5a@SE=$ANCHOR_SE to complete..."
    $PYTHON -m scripts.ablation_chain wait --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 --poll 60
    # Exclude pa=0.5 (degenerate point) from the mechanism search.
    BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 --exclude-pa 0.5 \
        --metric-key best_top1 --maximize)
    if [ -z "$BEST_PA" ]; then
        echo "ERROR: could not determine BEST_PA from 5a@SE=$ANCHOR_SE"; exit 1
    fi
fi
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

# ── Barrier on 5b at the chosen anchor (need best_β) ──────────────────────
if [ -z "$BEST_BETA" ] && [ -f "$OUTDIR_BASE/_best_beta.txt" ]; then
    BEST_BETA=$(cat "$OUTDIR_BASE/_best_beta.txt" | tr -d '[:space:]')
    echo "[5c] Read BEST_BETA=$BEST_BETA from _best_beta.txt"
fi
if [ -z "$BEST_BETA" ]; then
    echo "[5c] Waiting for 5b@SE=$ANCHOR_SE to complete (β sweep at pa=$BEST_PA)..."
    $PYTHON -m scripts.ablation_chain wait --phase 5b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 --poll 60
    BEST_BETA=$($PYTHON -m scripts.ablation_chain best --phase 5b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 \
        --metric-key best_top1 --maximize)
    if [ -z "$BEST_BETA" ]; then
        echo "ERROR: could not determine BEST_BETA from 5b@SE=$ANCHOR_SE"; exit 1
    fi
fi
# Persist best_β for rtx5090_6 faithful (idempotent — every GPU writes same value)
echo "$BEST_BETA" > "$OUTDIR_BASE/_best_beta.txt"

echo ""
echo "============================================================"
echo " ImageNet-ViT mini-ablation 5c — SE sweep @ pa=$BEST_PA  β=$BEST_BETA"
echo " ANCHOR_SE=$ANCHOR_SE (anchor's SE reuses 5a/5b cell)"
echo " (uniform K=46 + per-cat + wr=1.0 cosine, warmup_unit=epoch)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_se() {
    local SE=$1 SEED=$2
    local ENAME="phase5c_pa${BEST_PA}_beta${BEST_BETA}_se${SE}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    # Reuse anchor cell: SE == ANCHOR_SE → read from 5a (if β=1) or 5b (else).
    if [ "$SE" = "$ANCHOR_SE" ]; then
        if [ "$BEST_BETA" = "1.0" ]; then
            local sib="${OUTDIR_BASE}/phase5a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}/results.json"
        else
            local sib="${OUTDIR_BASE}/phase5b_pa${BEST_PA}_beta${BEST_BETA}${SE_SUFFIX}_s${SEED}/results.json"
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
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['SE'])

K = NUM_SUPERCLASSES
TOTAL_FFN = 384 * 4

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=TOTAL_FFN, se_ratio=SE_RATIO, num_branches=K)
branch_hidden = total_branch_ffn // K
fracs = compute_category_fractions('./data_cache/imagenet', num_categories=K)
assert abs(sum(fracs) - 1.0) < 1e-4

print(f'[cfg-gen] pa={PA} β={BETA} SE_ratio={SE_RATIO} warmup_unit=epoch K={K}')
print(f'[cfg-gen] branch_hidden={branch_hidden}  SE_dim={se_dim}  '
      f'effective_total={K*branch_hidden + se_dim} (dense={TOTAL_FFN})')

cfg = {
    'model': {
        'type': 'multi_branch_vit_small', 'num_classes': 1000,
        'num_branches': K,
        'branch_hidden': branch_hidden,
        'shared_expert_dim': se_dim,
        'img_size': 224, 'patch_size': 16,
        'drop_path_rate': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': BETA,
                   'warmup_schedule': 'cosine',
                   'warmup_unit': 'epoch'},
    'training': {'epochs': 100, 'batch_size': 256, 'grad_accum_steps': 1,
                  'lr': 2.5e-4, 'optimizer': 'adamw', 'weight_decay': 0.05,
                  'betas': [0.9, 0.999],
                  'lr_schedule': 'cosine', 'warmup_epochs': 5,
                  'label_smoothing': 0.1,
                  'mixup_alpha': 0.8, 'cutmix_alpha': 1.0, 'mixup_switch_prob': 0.5,
                  'amp_dtype': 'bf16', '_compile_mode': 'reduce-overhead',
                  'max_grad_norm': 1.0},
    'data': {'dataset': 'imagenet', 'data_dir': './data_cache/imagenet',
              'num_workers': 16, 'prefetch_factor': 4, 'augmentation': 'deit',
              'train_subset_frac': 0.2, 'train_subset_seed': 42},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[5c SE-sweep] seed=$SEED  pa=$BEST_PA β=$BEST_BETA ANCHOR_SE=$ANCHOR_SE"
    for SE in 0 0.5 1.0 2.0; do
        run_se "$SE" "$SEED"
    done
done

# ── Cross-GPU barrier on 5c → persist best_SE for rtx5090_6 ───────────────
# Every GPU arrives here after finishing its own seed's SE sweep. They all
# wait for the other GPUs, then compute the SAME best_SE deterministically
# and write the marker (idempotent — same value across GPUs).
echo ""
echo "[5c] Waiting for all 3 seeds' SE sweeps to complete..."
$PYTHON -m scripts.ablation_chain wait --phase 5c --base "$OUTDIR_BASE" \
    --seeds 42,123,456 --se-values 0,0.5,1.0,2.0 \
    --best-pa "$BEST_PA" --best-beta "$BEST_BETA" \
    --anchor-se "$ANCHOR_SE" --phase-prefix 5 --poll 60
BEST_SE=$($PYTHON -m scripts.ablation_chain best --phase 5c --base "$OUTDIR_BASE" \
    --seeds 42,123,456 --se-values 0,0.5,1.0,2.0 \
    --best-pa "$BEST_PA" --best-beta "$BEST_BETA" \
    --anchor-se "$ANCHOR_SE" --phase-prefix 5 \
    --metric-key best_top1 --maximize)
if [ -z "$BEST_SE" ]; then
    echo "ERROR: could not determine BEST_SE from 5c"; exit 1
fi
echo "$BEST_SE" > "$OUTDIR_BASE/_best_se.txt"
echo "[5c] Persisted _best_se.txt = $BEST_SE"

echo ""
echo "============================================================"
echo " 5c cross-GPU summary (SE sweep @ pa=$BEST_PA β=$BEST_BETA)"
echo "============================================================"
BEST_PA=$BEST_PA BEST_BETA=$BEST_BETA ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_vit_mini_ablation'
best_pa = os.environ['BEST_PA']; best_beta = os.environ['BEST_BETA']
anchor = os.environ['ANCHOR_SE']; suf = os.environ['SE_SUFFIX']
print(f"All runs: pa={best_pa} β={best_beta} per-cat epoch-warmup uniform K=46, ANCHOR_SE={anchor}.\n")
print(f"{'SE':>4}  {'mean top1':>11}  {'std':>6}  n")
for se in ('0', '0.5', '1.0', '2.0'):
    t1s = []
    for s in (42, 123, 456):
        if se == anchor:
            if best_beta == '1.0':
                p = os.path.join(base, f'phase5a_pa{best_pa}{suf}_s{s}', 'results.json')
            else:
                p = os.path.join(base, f'phase5b_pa{best_pa}_beta{best_beta}{suf}_s{s}', 'results.json')
        else:
            p = os.path.join(base, f'phase5c_pa{best_pa}_beta{best_beta}_se{se}_s{s}', 'results.json')
        if os.path.exists(p):
            t1s.append(json.load(open(p))['best_top1'])
    if t1s:
        m = sum(t1s)/len(t1s)
        sd = (sum((x-m)**2 for x in t1s)/len(t1s))**0.5 if len(t1s) > 1 else 0.0
        print(f"{se:>4}  {m:>11.3f}  {sd:>6.3f}  {len(t1s)}")

EOF
echo ""
echo "Final ours config (ViT, epoch warmup, ANCHOR_SE=$ANCHOR_SE):"
echo "  pa=$BEST_PA  β=$BEST_BETA  SE=$BEST_SE  warmup_unit=epoch"
echo "  Markers: $OUTDIR_BASE/_best_{pa,beta,se}.txt + _anchor_se.txt"
echo "  scripts/experiments/vit/main_table.sh will auto-read these."
echo ""
echo "Done: $(date)"
