#!/bin/bash
source "$(dirname "$0")/_ensure_data.sh"
###############################################################################
# RTX 5090 Script 8a: LoRA post-train mini-ablation — pa sweep at anchor.
#
# Mirror of ViT 5a for the LoRA / SuperNI track. Uses 20% SuperNI subset
# (fixed seed 42 stratified) for fast ablation — each run is ~1h on 5090
# instead of ~5h full. Main table (rtx5090_9) uses full data.
#
# Anchor config (fixed across 8a/8b/8c): K=20 Wang 2022 Domain clusters,
# MultiBranch-LoRA (r=16, α=32, all 7 linear targets), per-cat routing
# (β=1 in 8a), wr=1.0 cosine, warmup_unit=step (matches per-step cosine
# LR — design principle from trainer_lora.py).
#
# SE anchor = 1.0 (same choice as ViT 5a; NLP-derived default for
# shared-expert regime where per-cat has headroom).
#
# Sweep: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa, 3 seeds = 18 runs.
# Selection: strict argmax on 3-seed mean eval_rouge_l (Wang 2022 Tk-Instruct
# canonical metric for SuperNI). Excludes pa=0.5 as structural degenerate
# (g = pa - pi = 0 → per-cat inactive ∀β). Tie-break: smaller pa wins.
#
# eval_rouge_l in results.json is computed PER EPOCH inside trainer_lora.py
# (Wang 2022 / FLAN convention) and the top-level value = best epoch's
# ROUGE-L (== ROUGE on the saved best.pt checkpoint). best.pt selection
# also uses argmax ROUGE-L. Single metric, three uses — no protocol
# mismatch.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_lora_ablation"
DEVICE="cuda"
ANCHOR_SE=${ANCHOR_SE:-1.0}
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# 3-tier skip / auto-rerun-rouge logic (see scripts/_ensure_cell.sh).
source scripts/_ensure_cell.sh
if [ -n "$PA_VALUES" ]; then
    PA_LIST=($(echo "$PA_VALUES" | tr ',' ' '))
else
    PA_LIST=(0.5 0.6 0.7 0.8 0.9 1.0)
fi

if [ "$ANCHOR_SE" = "0" ]; then
    SE_SUFFIX=""
else
    SE_SUFFIX="_se${ANCHOR_SE}"
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " LoRA mini-ablation 8a — pa sweep (ANCHOR_SE=${ANCHOR_SE})"
echo " (K=20 Wang 2022 domains + per-cat β=1 + wr=1.0 cosine step)"
echo " ${#SEEDS[@]} seed(s), 20% SuperNI subset, $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phase8a_pa${PA}${SE_SUFFIX}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && return
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI ANCHOR_SE=$ANCHOR_SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_lora_rank_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
SE_RATIO = float(os.environ['ANCHOR_SE'])

K = 20
TOTAL_RANK = K * 16  # 320 — our no-SE baseline
r_expert, r_SE = split_lora_rank_budget_for_se(TOTAL_RANK, SE_RATIO, K)
print(f'[cfg-gen] pa={PA} β=1.0 SE={SE_RATIO} → r_expert={r_expert} r_SE={r_SE} '
      f'(total_rank = {K*r_expert + r_SE} vs target {TOTAL_RANK})')

cfg = {
    'model': {
        'type': 'multi_branch_lora',
        'base_model_name': 'meta-llama/Llama-3.2-1B',
        'num_experts': K,
        'rank': r_expert,
        'alpha': 2.0 * r_expert,
        'dropout': 0.05,
        'shared_expert_rank': r_SE,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
        'torch_dtype': 'bfloat16',
        'attn_implementation': 'sdpa',
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': PA, 'p_inactive': PI,
        'assignment': 'round_robin',
        'warmup_ratio': 1.0,
        'warmup_schedule': 'cosine',
        'warmup_unit': 'step',
        'amplification_beta': 1.0,
    },
    'training': {
        'epochs': 3, 'batch_size_per_device': 8, 'grad_accum_steps': 16,
        'lr': 2.0e-4, 'warmup_ratio_lr': 0.03, 'weight_decay': 0.0,
        'max_grad_norm': 1.0, 'max_seq_len': 1024, 'amp_dtype': 'bf16',
        'log_interval': 20, 'eval_interval_epochs': 1,
    },
    'data': {
        'data_root': './data_cache/lora/natural-instructions',
        'num_workers': 4, 'num_clusters': 20,
        'subset_frac_train': 0.2,   # 20% ablation subset per user directive
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
    echo "[8a pa-sweep ANCHOR_SE=$ANCHOR_SE PA_LIST=${PA_LIST[*]}] seed=$SEED"
    for PA in "${PA_LIST[@]}"; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " 8a seed-local summary (ANCHOR_SE=$ANCHOR_SE) — ROUGE-L"
echo "============================================================"
ANCHOR_SE=$ANCHOR_SE SE_SUFFIX=$SE_SUFFIX $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_lora_ablation'
suf = os.environ['SE_SUFFIX']
print(f"{'pa':>4}  {'mean rouge_l':>14}  {'std':>6}  n")
for pa in ('0.5', '0.6', '0.7', '0.8', '0.9', '1.0'):
    vals = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phase8a_pa{pa}{suf}_s{s}', 'results.json')
        if os.path.exists(p):
            v = json.load(open(p)).get('eval_rouge_l')
            if v is not None:
                vals.append(v)
    if vals:
        m = sum(vals)/len(vals)
        sd = (sum((x-m)**2 for x in vals)/len(vals))**0.5 if len(vals) > 1 else 0.0
        print(f"{pa:>4}  {m:>14.4f}  {sd:>6.4f}  {len(vals)}")
print(f"\n[8a ANCHOR_SE={os.environ['ANCHOR_SE']}] strict argmax on eval_rouge_l "
      f"(exclude pa=0.5 mech-OFF + pa=1.0 hard-routing degenerate) computed by "
      f"8b via scripts.ablation_chain when all 3 seeds land.")
EOF
echo "Done: $(date)"
