#!/bin/bash
source "$(dirname "$0")/_ensure_data.sh"
###############################################################################
# RTX 5090 Script 9: LoRA post-train faithful main table.
#
# 5 baselines + ours = 6 methods × 3 seeds = 18 main runs on full SuperNI.
# Method #6b (MB-LoRA no-routing + SE=best_SE, matched-SE baseline) is kept
# in the plan but ONLY run when 8c determines BEST_SE > 0 — see footer.
#
# Methods + native K config (each r adjusted to ≈ 225M total LoRA params
# on GQA-aware Llama-3.2-1B):
#   1. Single LoRA r=16           (Hu 2022 PEFT baseline, 11M — not matched)
#   2. Single LoRA r=320           (capacity-matched upper, 225M)
#   3. LoRAMoE K=6, r=76           (Dou 2023, FFN-only, 225M)
#   4. HydraLoRA N=8, r=67         (Tian 2024, asymmetric, 226M)
#   5. MoCLE E=4 + uni, r=63       (Gou 2024, cluster-cond, 225M)
#   6. MB-LoRA no-routing K=20, r=16, SE=0  (arch-matched baseline, 225M)
#   7. Ours (MB-LoRA + SoftSpecDrop K=20)   (reads _best_{pa,beta,se}.txt)
#
# Param-budget design: all baselines forced to 225M via r adjustment, BUT
# each retains its paper-native K (faithful to original method). See configs
# for exact derivations.
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_lora_faithful"
MINI_BASE=${MINI_BASE:-"./outputs/rtx5090_lora_ablation"}
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# 3-tier skip / auto-rerun-rouge logic (see scripts/_ensure_cell.sh).
source scripts/_ensure_cell.sh

# Read (BEST_PA, BEST_BETA, BEST_SE) from 8c markers. Env override > marker > fail.
_read_marker() {
    local var_name=$1 marker=$2 label=$3
    if [ -n "${!var_name}" ]; then
        echo "[rtx5090_9] $label=${!var_name} (from env)"; return
    fi
    if [ -f "$MINI_BASE/$marker" ]; then
        local v=$(cat "$MINI_BASE/$marker" | tr -d '[:space:]')
        if [ -n "$v" ]; then
            eval "$var_name=$v"
            echo "[rtx5090_9] $label=$v (from $MINI_BASE/$marker)"; return
        fi
    fi
    echo "ERROR: $label not set. Either export $var_name=... or run 8a→8b→8c first."
    exit 1
}
_read_marker BEST_PA   _best_pa.txt   BEST_PA
_read_marker BEST_BETA _best_beta.txt BEST_BETA
_read_marker BEST_SE   _best_se.txt   BEST_SE
BEST_PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " LoRA FAITHFUL main table (5 baselines + ours × ${#SEEDS[@]} seed(s))"
echo " OURS lock-in: pa=$BEST_PA β=$BEST_BETA SE=$BEST_SE step-warmup"
echo " $(date)"
echo "============================================================"

run_method_from_config() {
    local NAME=$1 CFG=$2
    for SEED in "${SEEDS[@]}"; do
        local ENAME="${NAME}_s${SEED}"
        local ODIR="${OUTDIR_BASE}/${ENAME}"
        ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && continue
        mkdir -p "$ODIR"
        echo "  Running $ENAME ... ($(date))"
        # Copy config, overwrite seed + output_dir.
        $PYTHON - <<PYEOF
import yaml, os
cfg = yaml.safe_load(open('$CFG'))
cfg['seed'] = $SEED
cfg['output_dir'] = '$ODIR'
cfg['experiment_name'] = '$ENAME'
yaml.dump(cfg, open('$ODIR/_tmp.yaml', 'w'))
PYEOF
        $PYTHON run_lora.py --wandb --config "$ODIR/_tmp.yaml" --device $DEVICE 2>&1 \
            | tee "$ODIR/$ENAME.log"
        echo "  $ENAME finished: $(date)"
    done
}

# [1] Single LoRA r=320 (capacity-matched upper)
echo "[1/8] Single LoRA r=320 (capacity-matched upper)"
run_method_from_config "single_lora_r320" "configs/lora/single_lora_r320.yaml"

# [2] LoRAMoE K=6 (Dou 2023 native, r=76 → 225M on Llama-3.2-1B)
echo "[2/8] LoRAMoE K=6 (Dou 2023)"
run_method_from_config "loramoe_k6" "configs/lora/loramoe.yaml"

# [3] HydraLoRA N=8 (Tian 2024 native, r=67 → 225M on Llama-3.2-1B)
echo "[3/8] HydraLoRA N=8 (Tian 2024)"
run_method_from_config "hydra_lora_n8" "configs/lora/hydra_lora.yaml"

# [4] MoCLE E=4+1universal (Gou 2024 native, r=63 → 225M on Llama-3.2-1B)
echo "[4/8] MoCLE E=4+universal (Gou 2024)"
run_method_from_config "mocle_e4" "configs/lora/mocle.yaml"

# Order rationale (changed 2026-04-29): K-dependent no-routing baselines run
# AFTER ours, NOT before. If ours @ K=20 underperforms baselines and we
# decide to retry with a different K, the K-dependent mb_lora_no_routing[+SE]
# baselines would have to be re-run anyway. Running them AFTER ours saves
# ~16h × 2 = 32h of compute that would otherwise be sunk on the wrong K.
# Independent baselines (single_lora / loramoe / hydra / mocle) are
# K-independent and stay first.

# NOTE: Single LoRA r=16 (Hu 2022 PEFT reference) was dropped from the main
# table on 2026-04-24: its 11M LoRA params are NOT capacity-matched to the
# 225M budget of all other methods — it was only kept as a "PEFT default"
# reference point. Including it would conflate method comparison with
# capacity effect. configs/lora/single_lora_r16.yaml is preserved for
# anyone who wants the PEFT-default reference, but is no longer in the
# paper's main table.

# [5] OURS: MB-LoRA + SoftSpecDrop K=20 with ablation-locked (pa, β, SE).
echo "[5/8] OURS — MB-LoRA + SoftSpecDrop K=20, pa=$BEST_PA β=$BEST_BETA SE=$BEST_SE"
for SEED in "${SEEDS[@]}"; do
    ENAME="ours_s${SEED}"
    ODIR="${OUTDIR_BASE}/${ENAME}"
    ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && continue
    mkdir -p "$ODIR"
    echo "  Running $ENAME ... ($(date))"
    ODIR=$ODIR SEED=$SEED PA=$BEST_PA PI=$BEST_PI BETA=$BEST_BETA SE=$BEST_SE \
        $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_lora_rank_budget_for_se

K = 20
TOTAL_RANK = K * 16
SE_RATIO = float(os.environ['SE'])
r_expert, r_SE = split_lora_rank_budget_for_se(TOTAL_RANK, SE_RATIO, K)

cfg = {
    'model': {
        'type': 'multi_branch_lora',
        'base_model_name': 'meta-llama/Llama-3.2-1B',
        'num_experts': K, 'rank': r_expert, 'alpha': 2.0*r_expert,
        'dropout': 0.05, 'shared_expert_rank': r_SE,
        'target_modules': ['q_proj','k_proj','v_proj','o_proj',
                            'gate_proj','up_proj','down_proj'],
        'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa',
    },
    'algorithm': {
        'type': 'soft_specdrop',
        'p_active': float(os.environ['PA']), 'p_inactive': float(os.environ['PI']),
        'assignment': 'round_robin', 'warmup_ratio': 1.0,
        'warmup_schedule': 'cosine', 'warmup_unit': 'step',
        'amplification_beta': float(os.environ['BETA']),
    },
    'training': {
        'epochs': 3, 'batch_size_per_device': 8, 'grad_accum_steps': 16,
        'lr': 2.0e-4, 'warmup_ratio_lr': 0.03, 'weight_decay': 0.0,
        'max_grad_norm': 1.0, 'max_seq_len': 1024, 'amp_dtype': 'bf16',
        'log_interval': 20, 'eval_interval_epochs': 1,
    },
    'data': {
        'data_root': './data_cache/lora/natural-instructions',
        'num_workers': 4, 'num_clusters': 20, 'subset_frac_train': 1.0,
        'instances_per_task_train': 100, 'instances_per_task_eval': 100,
        'cluster_cache_dir': './data_cache/lora',
    },
    'output_dir': os.environ['ODIR'],
    'seed': int(os.environ['SEED']),
    'experiment_name': f"ours_s{os.environ['SEED']}",
}
with open(os.path.join(os.environ['ODIR'], '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    $PYTHON run_lora.py --wandb --config "$ODIR/_tmp.yaml" --device $DEVICE 2>&1 \
        | tee "$ODIR/$ENAME.log"
    echo "  $ENAME finished: $(date)"
done

# [6] MB-LoRA no-routing K=20, SE=0 (architecture-matched baseline) — runs AFTER
# ours so a possible K-rebalance (if ours underperforms baselines) doesn't sink
# this cell's compute. Independent of BEST_SE.
echo "[6/8] MB-LoRA no-routing K=20 r=16 (arch-matched baseline)"
run_method_from_config "mb_lora_no_routing" "configs/lora/mb_lora_no_routing.yaml"

# [6b] Matched-SE no-routing baseline — auto-runs ONLY when BEST_SE > 0.
# (When BEST_SE == 0, method #6b is identical to #6 at SE=0 and skipped to
# save compute.) This is the architecture+SE-matched control for ours: ours
# gets gain from {routing, SE}; #6 has neither; #6b has only SE → clean
# attribution of routing contribution.
if [ "$BEST_SE" != "0" ] && [ "$BEST_SE" != "0.0" ]; then
    echo "[7/8] MB-LoRA no-routing + SE=$BEST_SE (matched-SE baseline, BEST_SE>0)"
    for SEED in "${SEEDS[@]}"; do
        ENAME="mb_lora_no_routing_se${BEST_SE}_s${SEED}"
        ODIR="${OUTDIR_BASE}/${ENAME}"
        ensure_cell_or_rerun_rouge "$ENAME" "$ODIR" "$PYTHON" && continue
        mkdir -p "$ODIR"
        echo "  Running $ENAME ... ($(date))"
        ODIR=$ODIR SEED=$SEED SE=$BEST_SE $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import split_lora_rank_budget_for_se
K = 20
TOTAL_RANK = K * 16
SE_RATIO = float(os.environ['SE'])
r_expert, r_SE = split_lora_rank_budget_for_se(TOTAL_RANK, SE_RATIO, K)
cfg = {
    'model': {
        'type': 'multi_branch_lora',
        'base_model_name': 'meta-llama/Llama-3.2-1B',
        'num_experts': K, 'rank': r_expert, 'alpha': 2.0*r_expert,
        'dropout': 0.05, 'shared_expert_rank': r_SE,
        'target_modules': ['q_proj','k_proj','v_proj','o_proj',
                            'gate_proj','up_proj','down_proj'],
        'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa',
    },
    'algorithm': {'type': 'no_dropout'},   # equal 1/K weights, NOT SoftSpecDrop
    'training': {
        'epochs': 3, 'batch_size_per_device': 8, 'grad_accum_steps': 16,
        'lr': 2.0e-4, 'warmup_ratio_lr': 0.03, 'weight_decay': 0.0,
        'max_grad_norm': 1.0, 'max_seq_len': 1024, 'amp_dtype': 'bf16',
        'log_interval': 20, 'eval_interval_epochs': 1,
    },
    'data': {
        'data_root': './data_cache/lora/natural-instructions',
        'num_workers': 4, 'num_clusters': 20, 'subset_frac_train': 1.0,
        'instances_per_task_train': 100, 'instances_per_task_eval': 100,
        'cluster_cache_dir': './data_cache/lora',
    },
    'output_dir': os.environ['ODIR'],
    'seed': int(os.environ['SEED']),
    'experiment_name': f"mb_lora_no_routing_se{os.environ['SE']}_s{os.environ['SEED']}",
}
with open(os.path.join(os.environ['ODIR'], '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
        $PYTHON run_lora.py --wandb --config "$ODIR/_tmp.yaml" --device $DEVICE 2>&1 \
            | tee "$ODIR/$ENAME.log"
        echo "  $ENAME finished: $(date)"
    done
else
    echo "[7/8] SKIPPED — BEST_SE=$BEST_SE, #6b would be identical to #6."
fi

# [8/8] APPENDIX ABLATION: LoRAMoE applied to ALL 7 linears (not just FFN-3).
# Dou 2023 natively specifies FFN-only, which confounds ours-vs-LoRAMoE: is
# LoRAMoE's worse performance from worse routing, or from adapting 3 sites
# instead of 7? This row isolates the routing effect by giving LoRAMoE the
# same 7-site coverage as ours / HydraLoRA / MoCLE.
#
# DISABLED: this variant OOMs
# at bs=8 on 32GB 5090 (K=6 LoRAs × 7 sites = 2.3× routing intermediate vs
# FFN-only). Was patched to bs=4/accum=32 in configs/lora/loramoe_all7.yaml
# but we skip it rather than reschedule for the appendix data; site-
# coverage confound is disclosed in the paper's baseline-adaptation appendix. Set
# RUN_LORAMOE_ALL7=1 to re-enable.
if [ "${RUN_LORAMOE_ALL7:-0}" = "1" ]; then
    echo "[8/8] APPENDIX — LoRAMoE K=6 all-7-linears (target-module ablation)"
    run_method_from_config "loramoe_k6_all7" "configs/lora/loramoe_all7.yaml"
else
    echo "[8/8] APPENDIX — LoRAMoE all-7-linears DISABLED (set RUN_LORAMOE_ALL7=1 to re-enable)"
fi

echo ""
echo "============================================================"
echo " LoRA FAITHFUL main-table summary"
echo "============================================================"
BEST_SE=$BEST_SE $PYTHON - <<'EOF'
import json, os
base = './outputs/rtx5090_lora_faithful'
best_se = os.environ.get('BEST_SE', '0')
methods = ['single_lora_r16','single_lora_r320','loramoe_k6','loramoe_k6_all7',
           'hydra_lora_n8','mocle_e4','mb_lora_no_routing']
labels  = ['SingleLoRA-r16','SingleLoRA-r320','LoRAMoE-K6-FFN','LoRAMoE-K6-all7',
           'HydraLoRA-N8','MoCLE-E4','MB-LoRA-NoRoute-SE0']
if best_se not in ('0', '0.0'):
    methods.append(f'mb_lora_no_routing_se{best_se}')
    labels.append(f'MB-LoRA-NoRoute-SE{best_se}')
methods.append('ours')
labels.append('Ours')
print(f"{'Method':20s}  {'CE loss':>14}  {'ROUGE-L':>10}  {'EM':>6}  n")
for m, l in zip(methods, labels):
    ce, rouge, em = [], [], []
    for s in (42, 123, 456):
        p = f'{base}/{m}_s{s}/results.json'
        if os.path.exists(p):
            d = json.load(open(p))
            ce.append(d.get('best_eval_loss'))
            if d.get('eval_rouge_l') is not None:
                rouge.append(d['eval_rouge_l'])
            if d.get('eval_exact_match') is not None:
                em.append(d['eval_exact_match'])
    def _fmt(vals, spec):
        if not vals:
            return '—'.rjust(14 if '>14' in spec else 10 if '>10' in spec else 6)
        vals = [v for v in vals if v is not None]
        if not vals:
            return '—'
        m = sum(vals)/len(vals)
        sd = (sum((x-m)**2 for x in vals)/len(vals))**0.5 if len(vals) > 1 else 0.0
        if '>14' in spec:
            return f'{m:.4f}±{sd:.4f}'
        return f'{m:.4f}'
    print(f"  {l:18s}  {_fmt(ce, '>14'):>14}  {_fmt(rouge, '>10'):>10}  {_fmt(em, '>6'):>6}  {len(ce)}")
EOF
echo "Done: $(date)"
