#!/bin/bash
###############################################################################
# RTX 5090 Script 3g: NLP mini-ablation — Phase G (wr sweep at BEST_PA_F,
# SE=0). Step 2 of CIFAR-optimal-defaults chain (F → G → H).
#
# Barriers on Phase F (18 runs = 6 pa × 3 seeds), picks BEST_PA_F, then
# sweeps wr ∈ {0.0, 0.2, 0.5} at (BEST_PA_F, SE=0). wr=1.0 is NOT re-run —
# it IS Phase F's BEST_PA_F result (free reference).
#
# Search: wr ∈ {0.0, 0.2, 0.5}, pa = BEST_PA_F (from Phase F), SE=0
# Fixed: K=7, ffn=220, no SE, 100M tokens, 10 ep
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseG_wr{wr}_pa{pa}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on Phase F ──────────────────────────────────────────────────────
echo "[Phase G] Waiting for Phase F completion across all 3 seeds..."
$PYTHON - <<'EOF'
import os, time, sys
base = 'outputs/rtx5090_nlp_mini_ablation'
pa_values = [('0.5','0.5'), ('0.6','0.4'), ('0.7','0.3'),
              ('0.8','0.2'), ('0.9','0.1'), ('1.0','0.0')]
all_seeds = (42, 123, 456)
while True:
    missing = [f'pa{pa}_s{s}' for pa, pi in pa_values for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase F: still {len(missing)}/18 missing')
    sys.stdout.flush()
    time.sleep(60)
print('  Phase F complete. Computing BEST_PA_F...')
EOF

if [ -n "$BEST_PA_F" ]; then
    echo "[Phase G] Using manual BEST_PA_F=$BEST_PA_F from env"
else
    BEST_PA_F=$($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_ppl = float('inf'); best_pa = None
for pa in ('0.5','0.6','0.7','0.8','0.9','1.0'):
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json')
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
if [ -z "$BEST_PA_F" ]; then
    echo "ERROR: could not determine BEST_PA_F from Phase F"; exit 1
fi
BEST_PI_F=$($PYTHON -c "print(round(1.0 - $BEST_PA_F, 2))")
echo ""
echo "============================================================"
echo " NLP mini-ablation Phase G — wr sweep at pa=$BEST_PA_F, SE=0 (${#SEEDS[@]} seeds)"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"
BASE_MODEL="'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': 220, 'dropout': 0.1"

run_wr() {
    local WR=$1 SEED=$2
    local ENAME="phaseG_wr${WR}_pa${BEST_PA_F}_s${SEED}"
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
    'algorithm': {'type': 'soft_specdrop', 'p_active': $BEST_PA_F, 'p_inactive': $BEST_PI_F, 'assignment': 'round_robin', 'warmup_ratio': $WR},
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
    echo "[Phase G] seed=$SEED"
    for WR in 0.0 0.2 0.5; do
        run_wr "$WR" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase G summary (wr sweep @ pa=$BEST_PA_F, SE=0)"
echo "============================================================"
BEST_PA_ENV="$BEST_PA_F" BEST_PI_ENV="$BEST_PI_F" $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA_ENV']; best_pi = os.environ['BEST_PI_ENV']
print(f"All runs use pa={best_pa} pi={best_pi} SE=0.")
print(f"{'wr':>5}  {'mean PPL':>10}  {'std':>6}  n seeds  {'note':<30}")

rows = []
# wr=1.0 reference from Phase F (free)
ppls = []
for s in (42, 123, 456):
    p = os.path.join(base, f'phaseF_pa{best_pa}_pi{best_pi}_s{s}', 'results.json')
    if os.path.exists(p):
        ppls.append(json.load(open(p))['best_val_ppl'])
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
    rows.append(('1.0', m, std, len(ppls)))
    print(f"{'1.0*':>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}  (ref: Phase F best_pa)")

for wr in ('0.0','0.2','0.5'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseG_wr{wr}_pa{best_pa}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
        rows.append((wr, m, std, len(ppls)))
        print(f"{wr:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}")

complete = [r for r in rows if r[3] == 3]
if complete:
    best = min(complete, key=lambda r: r[1])
    print(f"\nBEST_WR_G (3-seed mean): wr={best[0]} → {best[1]:.2f} PPL")
EOF
echo "Done: $(date)"
