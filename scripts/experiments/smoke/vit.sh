#!/bin/bash
###############################################################################
# Smoke Test — ImageNet × ViT-Small only (Setting 3, rtx5090_4).
#
# Phases:
#  1. Env check (CUDA + bf16)
#  2. ImageNet HF cache probe (warns if 150GB not pre-populated)
#  3. 6 faithful methods × 20 steps on 512-image subset
#  4. torch.compile mode comparison (ours method)
#  5. Time estimate for full 100-ep × 3-seed run
#
# Typical runtime: ~10-15 min after ImageNet cache exists. If HF cache is empty
# the first ViT method will block on a 150GB download — populate in advance.
###############################################################################

PYTHON=${PYTHON:-python}
set -e
SMOKE_OUT="./outputs/smoke_vit"
mkdir -p "$SMOKE_OUT"

echo "============================================================"
echo " RTX 5090 Smoke — ViT-Small ImageNet (Setting 3)"
echo " $(date)"
echo "============================================================"

# ─── 1. Env check ────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
assert torch.cuda.is_bf16_supported(), 'ViT setting uses bf16'
print(f'  CUDA: {torch.cuda.get_device_name(0)}  bf16: ON')
"

# ─── 2. ImageNet cache probe ─────────────────────────────────────────────────
IMAGENET_DIR="./data_cache/imagenet"
$PYTHON -c "
from pathlib import Path
cache = Path('$IMAGENET_DIR')
if not cache.exists() or not any(cache.rglob('*.arrow')):
    print('  ⚠ ImageNet cache empty at $IMAGENET_DIR')
    print('    Pre-populate with: huggingface-cli download ImageNet-1k')
    print('    HF will otherwise try a 150GB auto-fetch on first run.')
else:
    print(f'  ImageNet cache present at $IMAGENET_DIR')
"

# ─── 3. Per-method smoke ─────────────────────────────────────────────────────
echo ""
echo "=== 6 ViT methods × 20 steps on 512-image subset ==="

_run_cfg() {
    local NAME=$1 CFG=$2
    local ODIR="$SMOKE_OUT/${NAME}"
    rm -rf "$ODIR"; mkdir -p "$ODIR"
    $PYTHON -c "
import yaml
$CFG
cfg['output_dir'] = '$ODIR'
cfg['experiment_name'] = '$NAME'
cfg['seed'] = 42
yaml.dump(cfg, open('$ODIR/config.yaml', 'w'))
"
    local T0=$(date +%s)
    $PYTHON run.py --config "$ODIR/config.yaml" --device cuda > "$ODIR/run.log" 2>&1 \
        && STATUS="OK" || STATUS="FAIL"
    local T1=$(date +%s); local DT=$((T1 - T0))
    if grep -qE "Traceback|DEBUG.*NaN/Inf" "$ODIR/run.log" 2>/dev/null; then
        STATUS="ABNORMAL"
    fi
    printf "  %-22s %-9s  %3ds\n" "$NAME" "$STATUS" "$DT"
    echo "$DT" > "$ODIR/seconds"
}

TRAIN="'epochs':1,'batch_size':32,'grad_accum_steps':1,'lr':5e-4,'optimizer':'adamw','weight_decay':0.05,'lr_schedule':'cosine','warmup_epochs':0,'label_smoothing':0.1,'mixup_alpha':0.8,'cutmix_alpha':1.0,'mixup_switch_prob':0.5,'amp_dtype':'bf16','_no_compile':True,'max_steps':20"
DATA="'dataset':'imagenet','data_dir':'./data_cache/imagenet','num_workers':4,'augmentation':'deit','max_samples':512"

# vit_ours exercises every NEW-method code path that 5a/5b/5c/6 use:
#   • MultiBranchViT shared_expert_dim > 0  (SE block forward/backward)
#   • SoftSpecDrop frac_per_category + amplification_beta  (per-cat targets)
#   • warmup_schedule='cosine', warmup_unit='epoch'  (config wiring)
#   • split_ffn_budget_for_se(1536, 0.5, 46) + compute_category_fractions
# Using pa=0.6, β=4, SE=0.5 as a representative argmax for the paper's
# placeholder config — 5a/5b/5c will replace these with the actual argmax.
for BLK in \
    "vit_dense|cfg={'model':{'type':'vit_small','num_classes':1000,'drop_path_rate':0.1},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "vit_mod_squad|cfg={'model':{'type':'mod_squad_vit','num_classes':1000,'num_experts':16,'top_k':2,'mi_weight':0.001,'drop_path_rate':0.1},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "vit_soft_moe|cfg={'model':{'type':'soft_moe_vit','num_classes':1000,'num_experts':32,'slots_per_expert':1,'drop_path_rate':0.1},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "vit_comet|cfg={'model':{'type':'comet_vit','num_classes':1000,'p_keep':0.25,'drop_path_rate':0.1},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "vit_no_routing|cfg={'model':{'type':'multi_branch_vit_small','num_classes':1000,'num_branches':46,'drop_path_rate':0.1},'algorithm':{'type':'no_dropout'},'training':{$TRAIN},'data':{$DATA}}" \
    "vit_ours|
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se
_fracs = compute_category_fractions('./data_cache/imagenet', NUM_SUPERCLASSES)
_total_branch, _se_dim = split_ffn_budget_for_se(384*4, 0.5, 46)
_branch_hidden = _total_branch // 46
cfg={'model':{'type':'multi_branch_vit_small','num_classes':1000,'num_branches':46,'branch_hidden':_branch_hidden,'shared_expert_dim':_se_dim,'drop_path_rate':0.1},'algorithm':{'type':'soft_specdrop','p_active':0.6,'p_inactive':0.4,'assignment':'round_robin','warmup_ratio':1.0,'warmup_schedule':'cosine','warmup_unit':'epoch','frac_per_category':_fracs,'amplification_beta':4.0},'training':{$TRAIN},'data':{$DATA}}"
do
    _run_cfg "${BLK%%|*}" "${BLK#*|}"
done

# ─── 4. torch.compile comparison on ours ─────────────────────────────────────
echo ""
echo "=== ViT compile comparison (ours) ==="

_timed_compile() {
    local NAME=$1 CFG_BASE=$2
    for MODE in disabled default reduce-overhead; do
        local ODIR="$SMOKE_OUT/${NAME}_${MODE}"
        rm -rf "$ODIR"; mkdir -p "$ODIR"
        $PYTHON -c "
import yaml
$CFG_BASE
cfg['training']['_no_compile'] = ('$MODE' == 'disabled')
if '$MODE' != 'disabled': cfg['training']['_compile_mode'] = '$MODE'
cfg['output_dir'] = '$ODIR'
cfg['seed'] = 42
cfg['experiment_name'] = '${NAME}_${MODE}'
yaml.dump(cfg, open('$ODIR/config.yaml', 'w'))
"
        local T0=$(date +%s)
        $PYTHON run.py --config "$ODIR/config.yaml" --device cuda > "$ODIR/run.log" 2>&1 \
            && STATUS="OK" || STATUS="FAIL"
        local T1=$(date +%s); local DT=$((T1 - T0))
        if grep -qE "NaN|Inf|Traceback|RuntimeError" "$ODIR/run.log" 2>/dev/null; then
            STATUS="ABNORMAL"
        fi
        printf "  %-28s %-9s  %3ds\n" "${NAME}_${MODE}" "$STATUS" "$DT"
    done
}

VIT_OURS_CFG="
from data.imagenet import compute_category_fractions, NUM_SUPERCLASSES
from data.slimpajama import split_ffn_budget_for_se
_fracs = compute_category_fractions('./data_cache/imagenet', NUM_SUPERCLASSES)
_total_branch, _se_dim = split_ffn_budget_for_se(384*4, 0.5, 46)
_branch_hidden = _total_branch // 46
cfg={'model':{'type':'multi_branch_vit_small','num_classes':1000,'num_branches':46,'branch_hidden':_branch_hidden,'shared_expert_dim':_se_dim,'drop_path_rate':0.1},'algorithm':{'type':'soft_specdrop','p_active':0.6,'p_inactive':0.4,'assignment':'round_robin','warmup_ratio':1.0,'warmup_schedule':'cosine','warmup_unit':'epoch','frac_per_category':_fracs,'amplification_beta':4.0},'training':{$TRAIN},'data':{$DATA}}"
_timed_compile "vit_compile" "$VIT_OURS_CFG"

# ─── 5. Time estimate ────────────────────────────────────────────────────────
echo ""
echo "=== ViT full-run estimate (100ep × 1.28M × 3 seeds, rtx5090_4) ==="
$PYTHON -c "
def rs(p):
    try: return int(open(p).read().strip())
    except: return None
smoke = '$SMOKE_OUT'
total = 0
# 20 smoke steps × batch 32 on 512-sample subset → full 100ep × 1.28M/256 = 500K steps
# smoke-to-full scale ≈ 25000
for m in ['vit_dense','vit_mod_squad','vit_soft_moe','vit_comet','vit_no_routing','vit_ours']:
    s = rs(f'{smoke}/{m}/seconds')
    if s is None: continue
    per = s * 25000
    total += per * 3
    print(f'  {m:18s}: {s:3d}s smoke → per-seed {per/3600:5.1f}h')
print(f'  ViT 6 methods × 3 seeds, 3-GPU parallel: {total/3/3600:.1f}h')
"
echo ""
echo "============================================================"
echo " ViT smoke done: $(date)"
echo "============================================================"
