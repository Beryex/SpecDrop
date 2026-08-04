#!/bin/bash
###############################################################################
# RTX 5090 Script 3e: NLP mini-ablation — Phase E (warmup_ratio sweep WITH SE=1.0).
#
# Reads Phase D across 3 seeds (filesystem barrier), picks BEST_PA_SE, then
# sweeps wr ∈ {0.2, 0.5, 1.0} at (BEST_PA_SE, SE=1.0). wr=0.0 is NOT re-run —
# it IS Phase D's BEST_PA_SE result.
#
# Same rationale as Phase B for no-SE regime: CIFAR wr curve was near-flat,
# so 3 points suffice to verify NLP's wr landscape WITH SE present.
#
# Architecture: K=7, ffn=192, SE_dim=192 (same as Phase D and rtx5090_4's
# no_routing_se). Parameter-matched to dense (30.16M, +0.05%).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseE_wr{wr}_pa{pa}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier: wait for Phase D on ALL 3 seeds ────────────────────────────────
echo "[Phase E] Waiting for Phase D completion across all 3 seeds..."
$PYTHON - <<'EOF'
import os, time, sys
base = 'outputs/rtx5090_nlp_mini_ablation'
pa_values = [('0.5','0.5'), ('0.6','0.4'), ('0.7','0.3'),
              ('0.8','0.2'), ('0.9','0.1'), ('1.0','0.0')]
all_seeds = (42, 123, 456)
while True:
    missing = [f'pa{pa}_s{s}' for pa, pi in pa_values for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'phaseD_pa{pa}_pi{pi}_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase D: still {len(missing)}/18 missing')
    sys.stdout.flush()
    time.sleep(60)
print('  Phase D complete. Computing BEST_PA_SE...')
EOF

# Compute BEST_PA_SE. Allow env override for manual picking.
if [ -n "$BEST_PA_SE" ]; then
    echo "[Phase E] Using manual BEST_PA_SE=$BEST_PA_SE from env"
else
    BEST_PA_SE=$($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_ppl = float('inf'); best_pa = None
for pa in ('0.5','0.6','0.7','0.8','0.9','1.0'):
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseD_pa{pa}_pi{pi}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if len(ppls) == 3:
        m = sum(ppls) / 3
        if m < best_ppl:
            best_ppl = m
            best_pa = pa
print(best_pa)
EOF
)
fi
if [ -z "$BEST_PA_SE" ]; then
    echo "ERROR: could not determine BEST_PA_SE from Phase D"; exit 1
fi
BEST_PI_SE=$($PYTHON -c "print(round(1.0 - $BEST_PA_SE, 2))")
echo ""
echo "============================================================"
echo " NLP mini-ablation Phase E — wr sweep WITH SE=1.0 (${#SEEDS[@]} seeds)"
echo " Fixed: pa=$BEST_PA_SE pi=$BEST_PI_SE (from Phase D), SE=1.0"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"
BASE_MODEL="'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 192, 'shared_expert_dim': 192, 'dropout': 0.1"

run_wr() {
    local WR=$1 SEED=$2
    local ENAME="phaseE_wr${WR}_pa${BEST_PA_SE}_s${SEED}"
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
    'algorithm': {'type': 'soft_specdrop', 'p_active': $BEST_PA_SE, 'p_inactive': $BEST_PI_SE, 'assignment': 'round_robin', 'warmup_ratio': $WR},
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
    echo "[Phase E] seed=$SEED"
    for WR in 0.2 0.5 1.0; do
        run_wr "$WR" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase E summary"
echo "============================================================"
BEST_PA_ENV="$BEST_PA_SE" BEST_PI_ENV="$BEST_PI_SE" $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA_ENV']; best_pi = os.environ['BEST_PI_ENV']
print(f"All runs use pa={best_pa} pi={best_pi} SE=1.0.")
print(f"{'wr':>5}  {'mean PPL':>10}  {'std':>6}  n seeds  {'note':<30}")

# wr=0.0 ≡ Phase D BEST_PA_SE result
ppls = []
for s in (42, 123, 456):
    p = os.path.join(base, f'phaseD_pa{best_pa}_pi{best_pi}_s{s}', 'results.json')
    if os.path.exists(p):
        ppls.append(json.load(open(p))['best_val_ppl'])
rows = []
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
    rows.append(('0.0', m, std, len(ppls)))
    print(f"{'0.0':>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}  (= Phase D best_pa)")

for wr in ('0.2','0.5','1.0'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseE_wr{wr}_pa{best_pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
        rows.append((wr, m, std, len(ppls)))
        print(f"{wr:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}")

complete = [r for r in rows if r[3] == 3]
if complete:
    best = min(complete, key=lambda r: r[1])
    print(f"\nBEST_WR_SE (3-seed mean): wr={best[0]} → {best[1]:.2f} PPL")
    print(f"\nNLP with-SE optimum: pa={best_pa}, wr={best[0]}, SE_ratio=1.0")
EOF
echo "Done: $(date)"
