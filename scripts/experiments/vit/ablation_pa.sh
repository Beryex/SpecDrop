#!/bin/bash
###############################################################################
# RTX 5090 Script 5a: ImageNet ViT mini-ablation — pa sweep at Phase P anchor.
#
# Mirrors the NLP 3a phase for the ImageNet ViT-Small/16 setting. Uses per-cat
# routing (β=1 anchor) on top of the multi-branch ViT with K=46 BREEDS
# superclasses and a shared expert at the SE=1.0 anchor. pa warmup is cosine
# per-epoch (matches trainer.py's per-epoch scheduler.step() — NOT per-step
# like the NLP chain, per the "pa-warmup granularity matches LR schedule
# granularity" design principle).
#
# Anchor config (fixed across 5a/5b/5c): uniform branches + per-cat routing
# (β=1 in 5a) + wr=1.0 cosine warmup + warmup_unit=epoch + SE_ratio=1.0
# + BREEDS K=46 superclass labels.
#
# Unlike NLP (which had SE=0 as the primary anchor and SE=1.0 as fallback),
# ViT ablates at SE=1.0 directly — selected by the user to skip the fallback
# complexity now that NLP experience established shared-expert regime as the
# one where per-cat routing has headroom.
#
# Sweep: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa, 3 seeds each = 18 runs.
# Selection: strict argmax on 3-seed mean top1 (exact tie → smaller pa wins).
# No tie-break window.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_vit_mini_ablation"
DEVICE="cuda"
ANCHOR_SE=${ANCHOR_SE:-1.0}
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi
# PA_VALUES env: comma-separated subset of pa values to run. Default: full sweep.
if [ -n "$PA_VALUES" ]; then
    PA_LIST=($(echo "$PA_VALUES" | tr ',' ' '))
else
    PA_LIST=(0.5 0.6 0.7 0.8 0.9 1.0)
fi

# Dir-name suffix: empty only when ANCHOR_SE=0; every other value gets the suffix.
if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " ImageNet-ViT mini-ablation 5a — pa sweep @ Phase P anchor  (ANCHOR_SE=${ANCHOR_SE})"
echo " (uniform K=46 + per-cat β=1 + wr=1.0 cosine, warmup_unit=epoch)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phase5a_pa${PA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI ANCHOR_SE=$ANCHOR_SE $PYTHON - <<'PYEOF'
import os, yaml
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

K = NUM_SUPERCLASSES
TOTAL_FFN = 384 * 4  # ViT-S mlp_ratio × embed_dim = 1536

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=TOTAL_FFN, se_ratio=SE_RATIO, num_branches=K)
branch_hidden = total_branch_ffn // K
fracs = compute_category_fractions('./data_cache/imagenet', num_categories=K)
assert abs(sum(fracs) - 1.0) < 1e-4

print(f'[cfg-gen] pa={PA} β=1.0 SE_ratio={SE_RATIO} warmup_unit=epoch K={K}')
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
                   'frac_per_category': fracs, 'amplification_beta': 1.0,
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
    echo "[5a pa-sweep ANCHOR_SE=$ANCHOR_SE PA_LIST=${PA_LIST[*]}] seed=$SEED"
    for PA in "${PA_LIST[@]}"; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 5a seed-local summary (this GPU's seed completed, ANCHOR_SE=$ANCHOR_SE)"
echo "============================================================"
ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_vit_mini_ablation'
suf = os.environ['SE_SUFFIX']
anchor = os.environ['ANCHOR_SE']
print(f"{'pa':>4}  {'mean top1':>11}  {'std':>6}  n")
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    t1s = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase5a_pa{pa}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            t1s.append(json.load(open(p))['best_top1'])
    if t1s:
        m = sum(t1s)/len(t1s)
        sd = (sum((x-m)**2 for x in t1s)/len(t1s))**0.5 if len(t1s) > 1 else 0.0
        print(f"{pa:>4}  {m:>11.3f}  {sd:>6.3f}  {len(t1s)}")
print(f"\n[5a ANCHOR_SE={anchor}] strict argmax (exclude pa=0.5) will be "
      "computed by 5b via scripts.ablation_chain once all 3 seeds' runs land.")
EOF
echo "Done: $(date)"
