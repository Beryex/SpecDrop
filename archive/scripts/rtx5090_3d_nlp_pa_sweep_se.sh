#!/bin/bash
###############################################################################
# RTX 5090 Script 3d: NLP mini-ablation — Phase D (p_active sweep WITH SE=1.0).
#
# Parallel to rtx5090_3a (no-SE pa sweep) but with shared expert enabled at
# ratio 1.0. Motivated by Phase A finding: pa=0.5 (uniform/no-routing) was
# optimal without SE — category conditioning provided no benefit. Hypothesis
# for Phase D: SE's always-on shared backbone absorbs the shared cross-domain
# structure, freeing individual branches to specialize. If that's right, the
# inverted-U should reappear and optimum should shift away from pa=0.5.
#
# If Phase D optimum still at pa=0.5 → SE rescues performance via the shared
# backbone only; category conditioning remains useless on NLP at this scale.
# If Phase D optimum at pa>0.5 → SE + category conditioning are complementary;
# "SE enables specialization" becomes the paper story.
#
# Search: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa
# Fixed:  K=7, ffn=192, SE_dim=192 (SE ratio=1.0), wr=0.0, 100M tokens, 10 ep
# Param-matched: 7×192 + 192 = 1536 dense-equivalent (≈30.16M params, +0.05%)
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseD_pa{pa}_pi{pi}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase D — pa sweep WITH SE=1.0 (${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"
# WITH SE=1.0: ffn shrunk from 220→192, SE_dim=192 to keep 7×192+192=1536.
BASE_MODEL="'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 192, 'shared_expert_dim': 192, 'dropout': 0.1"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseD_pa${PA}_pi${PI}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    local TMPYAML="${ODIR}/_tmp.yaml"
    $PYTHON -c "
import yaml
cfg = {
    'model': {$BASE_MODEL},
    'algorithm': {'type': 'soft_specdrop', 'p_active': $PA, 'p_inactive': $PI, 'assignment': 'round_robin', 'warmup_ratio': 0.0},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
with open('$TMPYAML', 'w') as f:
    yaml.dump(cfg, f)
"
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "$TMPYAML" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase D] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase D summary (pa sweep WITH SE=1.0, once all 3 seeds done)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'pa':>5} {'pi':>5}  {'mean PPL':>10}  {'std':>6}  n seeds")
rows = []
for pa in ['0.5','0.6','0.7','0.8','0.9','1.0']:
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseD_pa{pa}_pi{pi}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls) / len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
complete = [r for r in rows if r[4] == 3]
if complete:
    best = min(complete, key=lambda r: r[2])
    print(f"\nBEST_PA_SE (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
    # Contrast with Phase A (no-SE) best
    pA = [r for r in rows if r[0] == '0.5']
    print(f"Compare Phase A pa=0.5 (no SE) at 100M: 56.74 PPL (reference).")
EOF
echo "Done: $(date)"
