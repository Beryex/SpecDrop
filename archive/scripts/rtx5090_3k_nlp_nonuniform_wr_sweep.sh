#!/bin/bash
###############################################################################
# RTX 5090 Script 3k: NLP mini-ablation — Phase K (wr sweep with NON-UNIFORM
# data-proportional branches at BEST_PA_J, SE=0).
#
# Step 2 of non-uniform chain J → K → L. Barriers on Phase J (6 pa × 3 seeds
# = 18 runs at wr=1.0, SE=0, non-uniform), picks BEST_PA_J, then sweeps
# wr ∈ {0.0, 0.2, 0.5} at (BEST_PA_J, SE=0). wr=1.0 is NOT re-run — it IS
# Phase J's BEST_PA_J result (free reference).
#
# Output: outputs/rtx5090_nlp_mini_ablation/nonuniform_pa{pa}_wr{wr}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on Phase J ──────────────────────────────────────────────────────
echo "[Phase K] Waiting for Phase J (18 non-uniform pa runs) ..."
$PYTHON - <<'EOF'
import os, time, sys
base = 'outputs/rtx5090_nlp_mini_ablation'
pa_values = ['0.5','0.6','0.7','0.8','0.9','1.0']
all_seeds = (42, 123, 456)
while True:
    missing = [f'pa{pa}_s{s}' for pa in pa_values for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase J: still {len(missing)}/18 missing'); sys.stdout.flush()
    time.sleep(60)
print('  Phase J complete. Computing BEST_PA_J ...')
EOF

if [ -n "$BEST_PA_J" ]; then
    echo "[Phase K] Using manual BEST_PA_J=$BEST_PA_J from env"
else
    BEST_PA_J=$($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_ppl = float('inf'); best_pa = None
for pa in ('0.5','0.6','0.7','0.8','0.9','1.0'):
    ppls = [json.load(open(os.path.join(base, f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json')))['best_val_ppl']
            for s in (42, 123, 456)]
    m = sum(ppls) / 3
    if m < best_ppl:
        best_ppl = m; best_pa = pa
print(best_pa)
EOF
)
fi
if [ -z "$BEST_PA_J" ]; then echo "ERROR: could not determine BEST_PA_J"; exit 1; fi
BEST_PI_J=$($PYTHON -c "print(round(1.0 - $BEST_PA_J, 2))")

echo ""
echo "============================================================"
echo " NLP mini-ablation Phase K — non-uniform wr sweep @ pa=$BEST_PA_J SE=0"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_wr() {
    local WR=$1 SEED=$2
    local ENAME="nonuniform_pa${BEST_PA_J}_wr${WR}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA_J PI=$BEST_PI_J WR=$WR $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import find_tokenize_cache, compute_proportional_ffn_dims

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
WR = float(os.environ['WR'])

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
ffn_dims, mapping = compute_proportional_ffn_dims(cache, total_ffn=1540, num_branches=7)
print(f'[cfg-gen] pa={PA} wr={WR} SE=0  ffn_dims={ffn_dims}')
assert mapping[0]['domain_name'] == 'RedPajamaCommonCrawl'
assert ffn_dims[0] == max(ffn_dims)

cfg = {
    'model': {'type': 'multi_branch_transformer_lm',
               'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
               'num_heads': 6, 'num_branches': 7,
               'ffn_dims_per_branch': ffn_dims, 'max_seq_len': 512, 'dropout': 0.1},
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': WR},
    'training': {'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
                  'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
                  'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'},
    'data': {'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
              'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase K] seed=$SEED"
    for WR in 0.0 0.2 0.5; do
        run_wr "$WR" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase K summary (non-uniform wr sweep @ pa=$BEST_PA_J SE=0)"
echo "============================================================"
BEST_PA_ENV="$BEST_PA_J" $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA_ENV']
print(f"All runs use pa={best_pa} SE=0 non-uniform. wr=1.0 from Phase J ref.")
print(f"{'wr':>5}  {'mean PPL':>10}  {'std':>6}  n  note")
rows = []

# wr=1.0 reference from Phase J
ppls = []
for s in (42, 123, 456):
    p = os.path.join(base, f'nonuniform_pa{best_pa}_wr1.0_s{s}', 'results.json')
    if os.path.exists(p):
        ppls.append(json.load(open(p))['best_val_ppl'])
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
    rows.append(('1.0', m, std, len(ppls)))
    print(f"{'1.0*':>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}  (Phase J ref)")

for wr in ('0.0','0.2','0.5'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'nonuniform_pa{best_pa}_wr{wr}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
        rows.append((wr, m, std, len(ppls)))
        print(f"{wr:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")

complete = [r for r in rows if r[3] == 3]
if complete:
    best = min(complete, key=lambda r: r[1])
    print(f"\nBEST_WR_K (3-seed mean): wr={best[0]} → {best[1]:.2f} PPL")
EOF
echo "Done: $(date)"
