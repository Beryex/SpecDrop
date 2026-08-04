#!/bin/bash
###############################################################################
# RTX 5090 Script 3f: NLP mini-ablation — Phase F (pa sweep at CIFAR-optimal
# defaults: wr=1.0, SE=0).
#
# Motivation. Phase A (wr=0, SE=0) found BEST_PA=0.5 ≡ no_routing. Phase D
# (wr=0, SE=1.0) found BEST_PA=0.6, within noise of 0.5 — SE shifted the
# curve down uniformly (−1.5 to −4.6 PPL) but did not shift pa ranking.
# Both phases used `lazy defaults` (wr=0, SE=0) when sweeping pa, rather
# than CIFAR-optimal defaults (wr=1.0, SE=0 for CIFAR Phase A).
#
# Phase F tests whether CIFAR-optimal warmup_ratio=1.0 changes pa ranking.
# Mechanism: at wr=1.0, all pa share the same initial forward pass
# (merged = mean(h_k) ≡ no_routing for any pa); they diverge only in late
# training as the cosine schedule ramps toward target (pa, pi). This
# mirrors CIFAR Phase A where wr=0 was the default (not CIFAR Phase A with
# wr scan, which was Phase B).
#
# Phase F uses CIFAR-optimal wr=1.0 and CIFAR-optimal SE=0 as fixed axes.
# Subsequent Phase G sweeps wr at BEST_PA_F (SE=0 fixed), Phase H sweeps
# SE at (BEST_PA_F, BEST_WR_G). Sequential ablation, CIFAR protocol exactly.
#
# Search: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa
# Fixed:  K=7, ffn=220, no SE, wr=1.0, 100M tokens, 10 ep
# Param-matched: 7×220 = 1540 ≈ 1536 dense-equivalent (−0.26%)
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseF_pa{pa}_pi{pi}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase F — pa sweep at wr=1.0, SE=0 (${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"
# No SE: ffn=220, no shared_expert_dim key (matches Phase A architecture).
BASE_MODEL="'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseF_pa${PA}_pi${PI}_s${SEED}"
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
    'algorithm': {'type': 'soft_specdrop', 'p_active': $PA, 'p_inactive': $PI, 'assignment': 'round_robin', 'warmup_ratio': 1.0},
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
    echo "[Phase F] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase F summary (pa sweep @ wr=1.0, SE=0)"
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
        p = os.path.join(base, f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls) > 1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
complete = [r for r in rows if r[4] == 3]
if complete:
    best = min(complete, key=lambda r: r[2])
    print(f"\nBEST_PA_F (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
    print(f"Compare Phase A pa=0.5 (wr=0, SE=0): 56.74 PPL")
    print(f"Compare Phase D pa=0.6 (wr=0, SE=1.0): 55.15 PPL")
EOF
echo "Done: $(date)"
