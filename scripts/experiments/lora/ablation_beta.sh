#!/bin/bash
source "$(dirname "$0")/_ensure_data.sh"
###############################################################################
# RTX 5090 Script 8b: LoRA mini-ablation — β sweep at (best_pa, SE=1.0 anchor).
#
# Mirror of ViT 5b for LoRA. Barriers on 8a@SE=1.0 (18 results.json), picks
# best_pa via strict argmax on eval_rouge_l (Wang 2022 Tk-Instruct canonical
# metric for SuperNI). TWO endpoints excluded by design:
#   - pa=0.5 (mech-OFF degenerate, g = pa - pi = 0 → per-cat inactive ∀β)
#   - pa=1.0 (mech-DEGENERATE: pi=0 → S=1, balanced-cat gap_c=1 → p_a^c=1,
#     p_i^c=0 = HARD routing. Soft fixed-denominator merge — SpecDrop's
#     core claim — collapses to deterministic top-1-by-assignment routing.
#     We exclude this endpoint so 8b/8c hyperparam search lands inside the
#     soft-merge regime where the method's mechanism actually applies.)
# Tie-break: smaller pa wins.
#
# Sweep: β ∈ {0, 1.0, 2.0, 4.0}. β=1.0 cell reuses 8a result (free reference).
# 3 new β × 3 seeds = 9 new runs.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_lora_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# 3-tier skip / auto-rerun-rouge logic (see scripts/_ensure_cell.sh).
source scripts/_ensure_cell.sh

mkdir -p "$OUTDIR_BASE"

if [ -z "$BEST_PA" ]; then
    ANCHOR_SE=${ANCHOR_SE:-1.0}
    echo "[8b] Waiting for 8a@SE=$ANCHOR_SE to complete (18 results.json)..."
    $PYTHON -m scripts.ablation_chain wait --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 --poll 60

    # Exclude pa=0.5 (mech-OFF, g=0) AND pa=1.0 (mech-degenerate, hard routing).
    BEST_PA=$($PYTHON -m scripts.ablation_chain best --phase 5a --base "$OUTDIR_BASE" \
        --seeds 42,123,456 --pa-values 0.5,0.6,0.7,0.8,0.9,1.0 \
        --anchor-se "$ANCHOR_SE" --phase-prefix 8 --exclude-pa 0.5,1.0 \
        --metric-key eval_rouge_l --maximize)
    if [ -z "$BEST_PA" ]; then
        echo "ERROR: could not determine BEST_PA from 8a@SE=$ANCHOR_SE"; exit 1
    fi
    echo "[8b] 8a@SE=$ANCHOR_SE strict argmax on eval_rouge_l (excluding pa=0.5,1.0) → best_pa = $BEST_PA"
fi

ANCHOR_SE=${ANCHOR_SE:-1.0}
PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

echo "$ANCHOR_SE" > "$OUTDIR_BASE/_anchor_se.txt"
echo "$BEST_PA" > "$OUTDIR_BASE/_best_pa.txt"

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

echo ""
echo "============================================================"
echo " LoRA mini-ablation 8b — β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
echo " (K=20 + per-cat + wr=1.0 cosine step, 20% subset)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_beta() {
    local BETA=$1 SEED=$2
    local ENAME="phase8b_pa${BEST_PA}_beta${BETA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && return
    # β=1.0 at (best_pa, ANCHOR_SE) IS the corresponding 8a cell; reuse.
    if [ "$BETA" = "1.0" ]; then
        local sibling_dir="${OUTDIR_BASE}/phase8a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}"
        if [ -f "$sibling_dir/results.json" ]; then
            echo "  $ENAME — reusing phase8a_pa${BEST_PA}${SE_SUFFIX}_s${SEED} (β=1 identical)"
            # Sibling may itself have missing ROUGE; ensure ROUGE is populated
            # there so the marker it leaves for 8c's argmin is complete.
            ensure_cell_or_rerun_rouge "phase8a_pa${BEST_PA}${SE_SUFFIX}_s${SEED}" "$sibling_dir" "$PYTHON" > /dev/null
            return
        fi
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$PI BETA=$BETA ANCHOR_SE=$ANCHOR_SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_lora_rank_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

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
    echo "[8b β-sweep] seed=$SEED  pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE"
    for BETA in 0 1.0 2.0 4.0; do
        run_beta "$BETA" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 8b seed-local summary (β sweep @ pa=$BEST_PA  ANCHOR_SE=$ANCHOR_SE) — ROUGE-L"
echo "============================================================"
BEST_PA=$BEST_PA ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_lora_ablation'
best_pa = os.environ['BEST_PA']; anchor = os.environ['ANCHOR_SE']; suf = os.environ['SE_SUFFIX']
print(f"{'β':>4}  {'mean rouge_l':>14}  {'std':>6}  n")
for beta in ('0', '1.0', '2.0', '4.0'):
    vals = []
    for s in (42, 123, 456):
        if beta == '1.0':
            p = os.path.join(base, f'phase8a_pa{best_pa}{suf}_s{s}', 'results.json')
        else:
            p = os.path.join(base, f'phase8b_pa{best_pa}_beta{beta}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            v = json.load(open(p)).get('eval_rouge_l')
            if v is not None:
                vals.append(v)
    if vals:
        m = sum(vals)/len(vals)
        sd = (sum((x-m)**2 for x in vals)/len(vals))**0.5 if len(vals) > 1 else 0.0
        print(f"{beta:>4}  {m:>14.4f}  {sd:>6.4f}  {len(vals)}")
EOF
echo "Done: $(date)"
