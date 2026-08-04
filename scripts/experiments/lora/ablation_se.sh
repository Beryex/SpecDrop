#!/bin/bash
source "$(dirname "$0")/_ensure_data.sh"
###############################################################################
# RTX 5090 Script 8c: LoRA mini-ablation — SE sweep at (best_pa, best_β).
#
# Mirror of ViT 5c. Barriers on 8b, reads best_β, sweeps SE_ratio ∈
# {0, 0.5, 1.0, 2.0}. At SE=anchor (1.0), reuses 8b's best-β cell.
# Net new: 3 SE × 3 seeds = 9 runs.
#
# Selection: strict argmax on 3-seed mean eval_rouge_l (Wang 2022 Tk-Instruct
# canonical). Tie-break: smaller SE wins. Same metric as 8a/8b — single
# selection criterion across the entire chain (8a → 8b → 8c → 9).
#
# Each SE ratio maps to (r_expert, r_SE) via split_lora_rank_budget_for_se
# with total_rank=320. Param budget stays ≈ 225M LoRA across all SE choices.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_lora_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# 3-tier skip / auto-rerun-rouge logic (see scripts/_ensure_cell.sh).
source scripts/_ensure_cell.sh

mkdir -p "$OUTDIR_BASE"

if [ -z "$ANCHOR_SE" ]; then
    if [ -f "$OUTDIR_BASE/_anchor_se.txt" ]; then
        ANCHOR_SE=$(cat "$OUTDIR_BASE/_anchor_se.txt" | tr -d '[:space:]')
    else
        ANCHOR_SE="1.0"
    fi
fi

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

if [ -z "$BEST_PA" ] && [ -f "$OUTDIR_BASE/_best_pa.txt" ]; then
    BEST_PA=$(cat "$OUTDIR_BASE/_best_pa.txt" | tr -d '[:space:]')
fi
if [ -z "$BEST_PA" ]; then
    echo "[8c] Need best_pa; barrier on 8a + compute..."
    $PYTHON -m scripts.ablation_chain wait --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 --poll 60
    BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 --exclude-pa 0.5,1.0 \
        --metric-key eval_rouge_l --maximize)
fi
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

if [ -z "$BEST_BETA" ] && [ -f "$OUTDIR_BASE/_best_beta.txt" ]; then
    BEST_BETA=$(cat "$OUTDIR_BASE/_best_beta.txt" | tr -d '[:space:]')
fi
if [ -z "$BEST_BETA" ]; then
    echo "[8c] Need best_β; barrier on 8b + compute..."
    $PYTHON -m scripts.ablation_chain wait --phase 5b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 --poll 60
    BEST_BETA=$($PYTHON -m scripts.ablation_chain best --phase 5b --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --beta-values 0,1.0,2.0,4.0 --best-pa "$BEST_PA" \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 \
        --metric-key eval_rouge_l --maximize)
fi
echo "$BEST_BETA" > "$OUTDIR_BASE/_best_beta.txt"

echo ""
echo "============================================================"
echo " LoRA mini-ablation 8c — SE sweep @ pa=$BEST_PA β=$BEST_BETA"
echo " ANCHOR_SE=$ANCHOR_SE, 20% subset, $(date)"
echo "============================================================"

run_se() {
    local SE=$1 SEED=$2
    local ENAME="phase8c_pa${BEST_PA}_beta${BEST_BETA}_se${SE}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && return
    if [ "$SE" = "$ANCHOR_SE" ]; then
        if [ "$BEST_BETA" = "1.0" ]; then
            local sib_dir="${OUTDIR_BASE}/phase8a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}"
        else
            local sib_dir="${OUTDIR_BASE}/phase8b_pa${BEST_PA}_beta${BEST_BETA}${SE_SUFFIX}_s${SEED}"
        fi
        if [ -f "$sib_dir/results.json" ]; then
            echo "  $ENAME — reusing $(basename $sib_dir)"
            # Ensure sibling has ROUGE too (idempotent no-op if already populated).
            ensure_cell_or_rerun_rouge "$(basename $sib_dir)" "$sib_dir" "$PYTHON" > /dev/null
            return
        fi
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BEST_BETA SE=$SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_lora_rank_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['SE'])

K = 20
TOTAL_RANK = K * 16
r_expert, r_SE = split_lora_rank_budget_for_se(TOTAL_RANK, SE_RATIO, K)
print(f'[cfg-gen] pa={PA} β={BETA} SE={SE_RATIO} → r_expert={r_expert} r_SE={r_SE}')

cfg = {
    'model': {
        'type': 'multi_branch_lora',
        'base_model_name': 'meta-llama/Llama-3.2-1B',
        'num_experts': K, 'rank': r_expert, 'alpha': 2.0*r_expert, 'dropout': 0.05,
        'shared_expert_rank': r_SE,
        'target_modules': ['q_proj','k_proj','v_proj','o_proj',
                            'gate_proj','up_proj','down_proj'],
        'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa',
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': PA, 'p_inactive': PI,
        'assignment': 'round_robin', 'warmup_ratio': 1.0,
        'warmup_schedule': 'cosine', 'warmup_unit': 'step',
        'amplification_beta': BETA,
    },
    'training': {
        'epochs': 3, 'batch_size_per_device': 8, 'grad_accum_steps': 16,
        'lr': 2.0e-4, 'warmup_ratio_lr': 0.03, 'weight_decay': 0.0,
        'max_grad_norm': 1.0, 'max_seq_len': 1024, 'amp_dtype': 'bf16',
        'log_interval': 20, 'eval_interval_epochs': 1,
    },
    'data': {
        'data_root': './data_cache/lora/natural-instructions',
        'num_workers': 4, 'num_clusters': 20, 'subset_frac_train': 0.2,
        'instances_per_task_train': 100, 'instances_per_task_eval': 100,
        'cluster_cache_dir': './data_cache/lora',
    },
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_lora.py --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[8c SE-sweep] seed=$SEED  pa=$BEST_PA  β=$BEST_BETA"
    for SE in 0 0.5 1.0 2.0; do
        run_se "$SE" "$SEED"
    done
done

# Cross-GPU barrier on 8c → persist best_SE.
echo ""
echo "[8c] Waiting for all 3 seeds' SE sweeps to complete..."
$PYTHON -m scripts.ablation_chain wait --phase 5c --base "$OUTDIR_BASE" \
    --seeds 42,123,456 --se-values 0,0.5,1.0,2.0 \
    --best-pa "$BEST_PA" --best-beta "$BEST_BETA" \
    --anchor-se "$ANCHOR_SE" --phase-prefix 8 --poll 60
BEST_SE=$($PYTHON -m scripts.ablation_chain best --phase 5c --base "$OUTDIR_BASE" \
    --seeds 42,123,456 --se-values 0,0.5,1.0,2.0 \
    --best-pa "$BEST_PA" --best-beta "$BEST_BETA" \
    --anchor-se "$ANCHOR_SE" --phase-prefix 8 \
    --metric-key eval_rouge_l --maximize)
if [ -z "$BEST_SE" ]; then
    echo "ERROR: could not determine BEST_SE from 8c"; exit 1
fi
echo "$BEST_SE" > "$OUTDIR_BASE/_best_se.txt"
echo "[8c] Persisted _best_se.txt = $BEST_SE"

echo ""
echo "Final LoRA ours config (from 8a → 8b → 8c):"
echo "  pa=$BEST_PA  β=$BEST_BETA  SE=$BEST_SE  warmup_unit=step"
echo "  Markers: $OUTDIR_BASE/_best_{pa,beta,se}.txt + _anchor_se.txt"
echo "  scripts/experiments/lora/main_table.sh reads these for the main table."
echo ""
echo "Done: $(date)"
