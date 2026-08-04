#!/bin/bash
###############################################################################
# RTX 5090 Script 5b: ImageNet ViT mini-ablation — β sweep at (best_pa, SE=1.0).
#
# Mirrors NLP 3b but tuned for ViT's simpler protocol: anchor SE is fixed at
# 1.0 (no degenerate-argmin fallback), so 5b only has to barrier on 5a and
# pick best_pa via strict argmax over top1, excluding pa=0.5 as the structural
# degenerate point.
#
# At pa=pi=0.5, g = p_a − p_i = 0, so gap_c = g·ratio = 0 for all categories
# and all β — the per-category mechanism is mathematically inactive regardless
# of β. We keep pa=0.5 runs as the β-invariant reference baseline but drop it
# from the "best method config" search (select_best_pa(exclude_pa=['0.5'])).
#
# Chain logic (runs AFTER 5a@SE=1.0):
#   1. Barrier on 5a@SE=1.0 (18 results.json).
#   2. best_pa = strict argmax @ SE=1.0 excluding pa=0.5.
#   3. Write "$OUTDIR_BASE/_anchor_se.txt" + "_best_pa.txt" (5c reads).
#   4. β sweep at (best_pa, SE=1.0). β=1.0 reuses 5a cell (free).
#
# ENV OVERRIDE: Set BEST_PA and ANCHOR_SE before running to skip auto-detect.
#
# Sweep: β ∈ {0, 1.0, 2.0, 4.0}. Net new runs: 3 β × 3 seeds = 9.
# (β=1.0 reuses 5a cell at best_pa, free reference.)
#
# β=0 is the scalar-per-cat reference: at β=0, ratio_c = ((1-frac_c)/(1-1/M))^0 = 1
# for all c, so gap_c = g (constant across categories). The per-category routing
# machinery is active (p_active_c, p_inactive_c differ from pa=0.5 uniform), but
# gets NO per-category differentiation. This isolates "pa > 0.5" (mechanism ON)
# from "β > 0" (category differentiation).
#
# Selection: strict argmax on top1 (exact tie → smaller β).
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_vit_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on 5a@SE=1.0, auto-detect best_pa ─────────────────────────────
if [ -z "$BEST_PA" ]; then
    ANCHOR_SE=${ANCHOR_SE:-1.0}
    echo "[5b] Waiting for 5a@SE=$ANCHOR_SE to complete (18 results.json files)..."
    $PYTHON -m scripts.ablation_chain wait --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 --poll 60

    # Exclude pa=0.5 (structural degenerate) from the mechanism argmax.
    BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 5 --exclude-pa 0.5 \
        --metric-key best_top1 --maximize)
    if [ -z "$BEST_PA" ]; then
        echo "ERROR: could not determine BEST_PA from 5a@SE=$ANCHOR_SE"; exit 1
    fi
    echo "[5b] 5a@SE=$ANCHOR_SE mechanism argmax (excluding pa=0.5) → best_pa = $BEST_PA"
fi

# Env-overridable (for manual re-runs / debugging)
ANCHOR_SE=${ANCHOR_SE:-1.0}
if [ -z "$BEST_PA" ]; then echo "ERROR: BEST_PA not set"; exit 1; fi
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

# Persist anchor decision for 5c (idempotent)
echo "$ANCHOR_SE" > "$OUTDIR_BASE/_anchor_se.txt"
echo "$BEST_PA" > "$OUTDIR_BASE/_best_pa.txt"

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

echo ""
echo "============================================================"
echo " ImageNet-ViT mini-ablation 5b — β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
echo " (uniform K=46 + per-cat + wr=1.0 cosine, warmup_unit=epoch)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_beta() {
    local BETA=$1 SEED=$2
    local ENAME="phase5b_pa${BEST_PA}_beta${BETA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    # β=1.0 at (best_pa, ANCHOR_SE) IS the corresponding 5a cell; reuse.
    if [ "$BETA" = "1.0" ]; then
        local sibling="${OUTDIR_BASE}/phase5a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}/results.json"
        if [ -f "$sibling" ]; then
            echo "  $ENAME — reusing phase5a_pa${BEST_PA}${SE_SUFFIX}_s${SEED} (β=1 identical)"
            return
        fi
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BETA ANCHOR_SE=$ANCHOR_SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

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
    echo "[5b β-sweep] seed=$SEED  pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
    for BETA in 0 1.0 2.0 4.0; do
        run_beta "$BETA" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 5b seed-local summary (β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE)"
echo "============================================================"
BEST_PA=$BEST_PA ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_vit_mini_ablation'
best_pa = os.environ['BEST_PA']; anchor = os.environ['ANCHOR_SE']; suf = os.environ['SE_SUFFIX']
print(f"All runs: pa={best_pa} SE={anchor} per-cat epoch-warmup uniform K=46.")
print(f"β=1.0 from 5a reference; β∈{{0,2,4}} are new runs.")
print(f"β=0: scalar per-cat reference (g>0 but no differentiation).\n")
print(f"{'β':>4}  {'mean top1':>11}  {'std':>6}  n")
for beta in ('0', '1.0', '2.0', '4.0'):
    t1s = []
    for s in (42, 123, 456):
        if beta == '1.0':
            p = os.path.join(base, f'phase5a_pa{best_pa}{suf}_s{s}', 'results.json')
        else:
            p = os.path.join(base, f'phase5b_pa{best_pa}_beta{beta}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            t1s.append(json.load(open(p))['best_top1'])
    if t1s:
        m = sum(t1s)/len(t1s)
        sd = (sum((x-m)**2 for x in t1s)/len(t1s))**0.5 if len(t1s) > 1 else 0.0
        print(f"{beta:>4}  {m:>11.3f}  {sd:>6.3f}  {len(t1s)}")
print(f"\n[5b ANCHOR_SE={anchor}] strict argmax will be computed by 5c.")
EOF
echo "Done: $(date)"
