#!/bin/bash
###############################################################################
# Smoke Test — CIFAR-100 × ResNet-110 only (Setting 1, rtx5090_2).
#
# Phases:
#  1. Env check
#  2. CIFAR-100 data download (auto, ~170MB, first run only)
#  3. faithful-method × 1 epoch real-data smoke
#  4. Time estimate for full 200-ep × 3-seed run
#
# First run: ~3-5 min. Subsequent: ~3 min.
###############################################################################

PYTHON=${PYTHON:-python}
set -e
SMOKE_OUT="./outputs/smoke_cifar"
mkdir -p "$SMOKE_OUT"

echo "============================================================"
echo " RTX 5090 Smoke — CIFAR (Setting 1)"
echo " $(date)"
echo "============================================================"

# ─── 1. Env check ────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
print(f'  CUDA: {torch.cuda.get_device_name(0)}  torch={torch.__version__}')
"

# ─── 2. CIFAR-100 readiness ──────────────────────────────────────────────────
echo ""
echo "=== CIFAR-100 data ==="
$PYTHON -c "
from data.cifar100 import ensure_downloaded as c_dl, validate_format as c_val
c_dl('./data_cache')
ok, msg = c_val('./data_cache'); assert ok, msg; print(f'  {msg}')
"

# ─── 3. Per-method smoke ─────────────────────────────────────────────────────
echo ""
echo "=== 6 CIFAR methods × 1 epoch (batch=128) ==="

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

TRAIN="'epochs':1,'batch_size':128,'lr':0.1,'momentum':0.9,'weight_decay':5e-4,'lr_schedule':'cosine','warmup_epochs':0"
DATA="'dataset':'cifar100','data_dir':'./data_cache','num_workers':4"

for BLK in \
    "resnet110|cfg={'model':{'type':'resnet','base_channels':16,'num_blocks':18,'num_classes':100},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "stoch_depth|cfg={'model':{'type':'resnet','base_channels':16,'num_blocks':18,'num_classes':100,'stochastic_depth_rate':0.5},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "et_dropout|cfg={'model':{'type':'example_tied_dropout_resnet110','num_classes':100,'num_examples':50000,'alpha_mem':0.5,'mem_keep_rate':0.5},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "ctx_dropout|cfg={'model':{'type':'contextual_dropout_resnet110','num_classes':100,'reduction':8},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "no_routing|cfg={'model':{'type':'multi_branch_resnet110','base_channels':16,'num_branches':20,'branch_channels':[4,7,14],'num_blocks':18,'num_classes':100},'algorithm':{'type':'no_dropout'},'training':{$TRAIN},'data':{$DATA}}" \
    "ours|cfg={'model':{'type':'multi_branch_resnet110','base_channels':16,'num_branches':20,'branch_channels':[4,7,14],'num_blocks':18,'num_classes':100},'algorithm':{'type':'soft_specdrop','p_active':0.7,'p_inactive':0.3,'assignment':'round_robin','warmup_ratio':1.0},'training':{$TRAIN},'data':{$DATA}}"
do
    _run_cfg "${BLK%%|*}" "${BLK#*|}"
done

# ─── 4. Time estimate ────────────────────────────────────────────────────────
echo ""
echo "=== CIFAR full-run estimate (200ep × 3 seeds, rtx5090_2) ==="
$PYTHON -c "
def rs(p):
    try: return int(open(p).read().strip())
    except: return None
smoke = '$SMOKE_OUT'
total = 0
for m in ['resnet110','stoch_depth','et_dropout','ctx_dropout','no_routing','ours']:
    s = rs(f'{smoke}/{m}/seconds')
    if s is None: continue
    per = s * 200
    total += per * 3
    print(f'  {m:18s}: {s:3d}s/ep → per-seed {per/3600:5.2f}h')
print(f'  CIFAR 6 methods × 3 seeds, 3-GPU parallel: {total/3/3600:.1f}h')
"
echo ""
echo "============================================================"
echo " CIFAR smoke done: $(date)"
echo "============================================================"
