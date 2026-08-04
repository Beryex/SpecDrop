#!/bin/bash
###############################################################################
# RTX 5090 Script 3l: NLP mini-ablation — Phase L (SE ratio sweep with
# NON-UNIFORM data-proportional branches at BEST_PA_J, BEST_WR_K).
#
# Step 3 of non-uniform chain J → K → L. Barriers on Phase J + K. For each
# SE ratio X the budget splits as:
#     total_branch_ffn = 1540 × K / (K + X)
#     SE_dim           = 1540 × X / (K + X)
#     ffn_dims[k]      = data-proportional within total_branch_ffn
# This keeps the dense-equivalent parameter budget constant at 30.14M across
# all SE ratios (within the ±2% sanity-check tolerance).
#
# For K=7, total=1540:
#     SE=0.5x : branches≈1437, SE≈103
#     SE=1.0x : branches≈1348, SE≈192
#     SE=4.0x : branches≈980,  SE≈560
# SE=0 reference comes from Phase J (if BEST_WR_K=1.0) or Phase K.
#
# Output: outputs/rtx5090_nlp_mini_ablation/nonuniform_se{X}_pa{pa}_wr{wr}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on Phase J + K ──────────────────────────────────────────────────
echo "[Phase L] Waiting for Phase J + K ..."
$PYTHON - <<'EOF'
import os, time, json, sys
base = 'outputs/rtx5090_nlp_mini_ablation'
pa_values = ['0.5','0.6','0.7','0.8','0.9','1.0']
all_seeds = (42, 123, 456)

# Wait for Phase J (18)
while True:
    missing = [f'pa{pa}_s{s}' for pa in pa_values for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase J: still {len(missing)}/18 missing'); sys.stdout.flush()
    time.sleep(60)

# Pick BEST_PA_J
best_ppl = float('inf'); best_pa = None
for pa in pa_values:
    ppls = [json.load(open(os.path.join(base, f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json')))['best_val_ppl']
            for s in all_seeds]
    m = sum(ppls)/3
    if m < best_ppl:
        best_ppl = m; best_pa = pa
print(f'  BEST_PA_J = {best_pa} (mean PPL {best_ppl:.2f})')

# Wait for Phase K (9)
while True:
    missing = [f'wr{wr}_s{s}' for wr in ('0.0','0.2','0.5') for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'nonuniform_pa{best_pa}_wr{wr}_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase K: still {len(missing)}/9 missing'); sys.stdout.flush()
    time.sleep(60)
print('  Phase J + K complete. Computing BEST_WR_K ...')
EOF

read BEST_PA BEST_WR <<< $($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'

# Phase J best_pa
best_ppl = float('inf'); best_pa = None
for pa in ('0.5','0.6','0.7','0.8','0.9','1.0'):
    ppls = [json.load(open(os.path.join(base, f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json')))['best_val_ppl']
            for s in (42, 123, 456)]
    m = sum(ppls)/3
    if m < best_ppl:
        best_ppl = m; best_pa = pa

# wr=1.0 from Phase J; wr<1.0 from Phase K
phaseK = {'1.0': best_ppl}
for wr in ('0.0','0.2','0.5'):
    ppls = [json.load(open(os.path.join(base, f'nonuniform_pa{best_pa}_wr{wr}_s{s}', 'results.json')))['best_val_ppl']
            for s in (42, 123, 456)]
    phaseK[wr] = sum(ppls)/3
best_wr = min(phaseK, key=phaseK.get)
print(f'{best_pa} {best_wr}')
EOF
)
if [ -z "$BEST_PA" ] || [ -z "$BEST_WR" ]; then
    echo "ERROR: could not determine BEST_PA / BEST_WR from Phases J+K"; exit 1
fi
BEST_PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")

echo ""
echo "============================================================"
echo " NLP mini-ablation Phase L — non-uniform SE sweep @ pa=$BEST_PA wr=$BEST_WR"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_se() {
    local X=$1 SEED=$2
    local ENAME="nonuniform_se${X}x_pa${BEST_PA}_wr${BEST_WR}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$BEST_PA PI=$BEST_PI WR=$BEST_WR SE_RATIO=$X $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import find_tokenize_cache, compute_proportional_ffn_dims, split_ffn_budget_for_se

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
WR = float(os.environ['WR']); SE_RATIO = float(os.environ['SE_RATIO'])

# Split total dense-equivalent budget between branches and SE.
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
print(f'[cfg-gen] SE_ratio={SE_RATIO}  total_branch_ffn={total_branch_ffn}  SE_dim={se_dim}')
print(f'[cfg-gen] branch_budget + SE_dim = {total_branch_ffn + se_dim} (target 1540)')

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
ffn_dims, mapping = compute_proportional_ffn_dims(
    cache, total_ffn=total_branch_ffn, num_branches=7)
print(f'[cfg-gen] ffn_dims (domain_id order) = {ffn_dims}  sum={sum(ffn_dims)}')
for m in mapping:
    print(f"  b{m['branch']}: {m['domain_name']:<30} {m['frac']:>7.2%}  ffn={m['ffn_dim']}")
assert mapping[0]['domain_name'] == 'RedPajamaCommonCrawl'
assert ffn_dims[0] == max(ffn_dims), "CC must still get the largest branch"

cfg = {
    'model': {'type': 'multi_branch_transformer_lm',
               'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
               'num_heads': 6, 'num_branches': 7,
               'ffn_dims_per_branch': ffn_dims,
               'shared_expert_dim': se_dim,
               'max_seq_len': 512, 'dropout': 0.1},
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
    echo "[Phase L] seed=$SEED"
    for X in 0.5 1.0 4.0; do
        run_se "$X" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase L summary (non-uniform SE sweep @ pa=$BEST_PA wr=$BEST_WR)"
echo "============================================================"
BEST_PA_ENV="$BEST_PA" BEST_PI_ENV="$BEST_PI" BEST_WR_ENV="$BEST_WR" $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA_ENV']; best_pi = os.environ['BEST_PI_ENV']; best_wr = os.environ['BEST_WR_ENV']
print(f"All runs use pa={best_pa} wr={best_wr} (non-uniform, data-proportional).")
print(f"{'SE ratio':>10}  {'mean PPL':>10}  {'std':>6}  n  note")

# SE=0 reference: Phase J if best_wr=1.0 else Phase K
if best_wr == '1.0':
    paths = [os.path.join(base, f'nonuniform_pa{best_pa}_wr1.0_s{s}', 'results.json') for s in (42, 123, 456)]
    ref_note = '(ref: Phase J best_pa)'
else:
    paths = [os.path.join(base, f'nonuniform_pa{best_pa}_wr{best_wr}_s{s}', 'results.json') for s in (42, 123, 456)]
    ref_note = f'(ref: Phase K wr={best_wr})'
ppls = [json.load(open(p))['best_val_ppl'] for p in paths if os.path.exists(p)]
rows = []
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
    print(f"{'0x*':>10}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}  {ref_note}")

for X in ('0.5x','1.0x','4.0x'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'nonuniform_se{X[:-1]}_pa{best_pa}_wr{best_wr}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
        rows.append((X, m, std, len(ppls)))
        print(f"{X:>10}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")

complete = [r for r in rows if r[3] == 3]
if complete:
    best = min(complete, key=lambda r: r[1])
    print(f"\nBEST_SE_L (3-seed mean): {best[0]} → {best[1]:.2f} PPL")
    print(f"\nFINAL OPTIMUM for NLP 100M (non-uniform chain J→K→L):")
    print(f"  pa={best_pa} wr={best_wr} SE_ratio={best[0]}")
    print(f"\nCompare to uniform chain (F→G→H):  BEST=no_routing+SE=1.0 (Phase D)=55.18 PPL")
EOF
echo "Done: $(date)"
