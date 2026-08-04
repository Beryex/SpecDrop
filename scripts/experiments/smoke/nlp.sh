#!/bin/bash
###############################################################################
# Smoke Test — SlimPajama × Transformer LM only (Setting 2, rtx5090_4).
#
# Phases:
#  1. Env check
#  2a. SlimPajama tokenize 500M (ONE-TIME, ~20-40 min; for rtx5090_4_nlp_faithful)
#  2b. SlimPajama tokenize 100M (ONE-TIME, ~4-8 min; for rtx5090_3 NLP mini-ablation)
#  3. 7 faithful methods × 1 epoch on 2M-token subset
#  4. torch.compile mode comparison (ours method; trainer_nlp hardcodes
#     compile=off by default, runs here for reference only)
#  5. Time estimate for full 10-ep × 500M × 3-seed run
#
# First run: ~45 min (tokenize dominates). Subsequent: ~10 min.
###############################################################################

PYTHON=${PYTHON:-python}
set -e
SMOKE_OUT="./outputs/smoke_nlp"
mkdir -p "$SMOKE_OUT"

echo "============================================================"
echo " RTX 5090 Smoke — NLP (Setting 2)"
echo " $(date)"
echo "============================================================"

# ─── 1. Env check ────────────────────────────────────────────────────────────
$PYTHON -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
print(f'  CUDA: {torch.cuda.get_device_name(0)}  bf16: {torch.cuda.is_bf16_supported()}')
"

# ─── 2a. SlimPajama pre-tokenize 500M (main runs: rtx5090_4_nlp_faithful) ────
echo ""
echo "=== SlimPajama tokenize 500M (one-time, for rtx5090_4) ==="
NLP_CACHE_DIR="./data_cache/slimpajama"
if ls "$NLP_CACHE_DIR"/tokenized_train_seq512_tok500000000_vocab*.pt >/dev/null 2>&1 \
   || ls "$NLP_CACHE_DIR"/tokenized_train_seq512_tok500000000_tok*.pt >/dev/null 2>&1; then
    echo "  500M cache present ✓ (accepts both _vocab{hash} and legacy _tok{hash} suffix)"
else
    echo "  Tokenizing 500M tokens (~20-40 min)..."
    $PYTHON -c "
from data.slimpajama import get_slimpajama_dataloaders
_ = get_slimpajama_dataloaders(
    data_dir='$NLP_CACHE_DIR', batch_size=4, num_workers=2,
    max_seq_len=512, max_train_tokens=500_000_000, max_val_tokens=10_000_000)
print('  ✓ 500M train + 10M val cached')
"
fi

# ─── 2b. SlimPajama pre-tokenize 100M (NLP mini-ablation: rtx5090_3a/3b/3c) ──
# The cache filename is tokenized_{split}_seq{seq_len}_tok{N}_vocab{hash}.pt
# so different max_train_tokens values need separate caches. The 3 mini-
# ablation scripts all use max_train_tokens=100_000_000, so pre-tokenize once
# here to avoid each of the 36 runs re-tokenizing.
# val cache (5M) is already produced as a side-effect of the 500M run above
# (same max_val_tokens default), so no additional val work needed.
echo ""
echo "=== SlimPajama tokenize 100M (one-time, for rtx5090_3 mini-ablation) ==="
if ls "$NLP_CACHE_DIR"/tokenized_train_seq512_tok100000000_vocab*.pt >/dev/null 2>&1; then
    echo "  100M cache present ✓"
else
    echo "  Tokenizing 100M tokens (~4-8 min)..."
    $PYTHON -c "
from data.slimpajama import get_slimpajama_dataloaders
_ = get_slimpajama_dataloaders(
    data_dir='$NLP_CACHE_DIR', batch_size=4, num_workers=2,
    max_seq_len=512, max_train_tokens=100_000_000, max_val_tokens=5_000_000)
print('  ✓ 100M train + 5M val cached')
"
fi

# ─── 3. Per-method smoke ─────────────────────────────────────────────────────
echo ""
echo "=== 7 NLP methods × 1 epoch on 2M subset ==="

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
    $PYTHON run_nlp.py --config "$ODIR/config.yaml" --device cuda > "$ODIR/run.log" 2>&1 \
        && STATUS="OK" || STATUS="FAIL"
    local T1=$(date +%s); local DT=$((T1 - T0))
    if grep -qE "Traceback|DEBUG.*NaN/Inf" "$ODIR/run.log" 2>/dev/null; then
        STATUS="ABNORMAL"
    fi
    printf "  %-22s %-9s  %3ds\n" "$NAME" "$STATUS" "$DT"
    echo "$DT" > "$ODIR/seconds"
}

TRAIN="'epochs':1,'batch_size':32,'max_tokens':2_000_000,'lr':3e-4,'optimizer':'adamw','weight_decay':0.1,'lr_schedule':'cosine','warmup_steps':50,'max_grad_norm':1.0"
DATA="'dataset':'slimpajama','data_dir':'./data_cache/slimpajama','num_workers':2,'max_seq_len':512,'max_train_tokens':2_000_000,'max_val_tokens':200_000"
LM="'vocab_size':50257,'hidden_dim':384,'num_layers':6,'num_heads':6,'max_seq_len':512"

for BLK in \
    "nlp_dense|cfg={'model':{'type':'transformer_lm',$LM,'ffn_dim':1536,'dropout':0.1},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "nlp_switch|cfg={'model':{'type':'switch_transformer_lm',$LM,'num_experts':32,'ffn_dim_per_expert':48,'load_balance_weight':0.01},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "nlp_hash|cfg={'model':{'type':'hash_layers_transformer_lm',$LM,'num_experts':8,'ffn_dim_per_expert':192},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "nlp_smoe|cfg={'model':{'type':'smoe_dropout_transformer_lm',$LM,'num_experts':16,'ffn_dim_per_expert':96,'k_init':1,'expert_drop_prob':0.0},'algorithm':{'type':'none'},'training':{$TRAIN},'data':{$DATA}}" \
    "nlp_demix|cfg={'model':{'type':'demix_transformer_lm',$LM,'num_domains':7,'ffn_dim_per_expert':220},'algorithm':{'type':'none'},'training':{$TRAIN,'demix_eval_mode':'mixture'},'data':{$DATA}}" \
    "nlp_no_routing|cfg={'model':{'type':'multi_branch_transformer_lm',$LM,'num_branches':7,'ffn_dim_per_branch':220},'algorithm':{'type':'no_dropout'},'training':{$TRAIN},'data':{$DATA}}" \
    "nlp_ours|cfg={'model':{'type':'multi_branch_transformer_lm',$LM,'num_branches':7,'ffn_dim_per_branch':220},'algorithm':{'type':'soft_specdrop','p_active':0.7,'p_inactive':0.3,'assignment':'round_robin','warmup_ratio':1.0},'training':{$TRAIN},'data':{$DATA}}"
do
    _run_cfg "${BLK%%|*}" "${BLK#*|}"
done

# ─── 4. torch.compile comparison on ours ─────────────────────────────────────
echo ""
echo "=== NLP compile comparison (ours — reference only; trainer hardcodes off) ==="

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
        $PYTHON run_nlp.py --config "$ODIR/config.yaml" --device cuda > "$ODIR/run.log" 2>&1 \
            && STATUS="OK" || STATUS="FAIL"
        local T1=$(date +%s); local DT=$((T1 - T0))
        if grep -qE "NaN|Inf|Traceback|RuntimeError" "$ODIR/run.log" 2>/dev/null; then
            STATUS="ABNORMAL"
        fi
        printf "  %-28s %-9s  %3ds\n" "${NAME}_${MODE}" "$STATUS" "$DT"
    done
}

NLP_OURS_CFG="cfg={'model':{'type':'multi_branch_transformer_lm',$LM,'num_branches':7,'ffn_dim_per_branch':220},'algorithm':{'type':'soft_specdrop','p_active':0.7,'p_inactive':0.3,'assignment':'round_robin','warmup_ratio':1.0},'training':{$TRAIN},'data':{$DATA}}"
_timed_compile "nlp_compile" "$NLP_OURS_CFG"

# ─── 5. Time estimate ────────────────────────────────────────────────────────
echo ""
echo "=== NLP full-run estimate (10ep × 500M × 3 seeds, rtx5090_3) ==="
$PYTHON -c "
def rs(p):
    try: return int(open(p).read().strip())
    except: return None
smoke = '$SMOKE_OUT'
total = 0
# 2M/500M = 1/250; 1ep/10ep = 1/10; so smoke-to-full factor = 2500
for m in ['nlp_dense','nlp_switch','nlp_hash','nlp_smoe','nlp_demix','nlp_no_routing','nlp_ours']:
    s = rs(f'{smoke}/{m}/seconds')
    if s is None: continue
    per = s * 2500
    total += per * 3
    print(f'  {m:18s}: {s:3d}s (2M, 1ep) → per-seed {per/3600:5.2f}h')
print(f'  NLP 7 methods × 3 seeds, 3-GPU parallel: {total/3/3600:.1f}h')
"
echo ""
echo "============================================================"
echo " NLP smoke done: $(date)"
echo "============================================================"
