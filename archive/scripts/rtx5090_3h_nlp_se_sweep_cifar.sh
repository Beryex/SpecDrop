#!/bin/bash
###############################################################################
# RTX 5090 Script 3h: NLP mini-ablation — Phase H (SE ratio sweep at
# BEST_PA_F, BEST_WR_G). Step 3 of CIFAR-optimal-defaults chain (F → G → H).
#
# Barriers on Phase F + G. Picks BEST_PA_F from Phase F, BEST_WR_G from
# Phase G (or Phase F at wr=1.0 if that remains optimal). Sweeps
# SE_ratio ∈ {0.5, 1.0, 4.0}. SE=0 is NOT re-run — it IS the reference
# from whichever of Phase F / Phase G holds the BEST_PA_F × BEST_WR_G cell.
#
# Param-matching: for each SE ratio X, branch_ffn = round(1536 / (K + X))
# and shared_expert_dim = round(1536 × X / (K + X)). With K=7:
#   SE=0.5x : ffn=205, SE=102  (+0.07% over dense)
#   SE=1.0x : ffn=192, SE=192  (+0.05%)
#   SE=4.0x : ffn=140, SE=559  (+0.10%)
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseH_se{X}x_pa{pa}_wr{wr}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"

# ── Barrier on Phase F + Phase G ────────────────────────────────────────────
echo "[Phase H] Waiting for Phase F + Phase G completion..."
$PYTHON - <<'EOF'
import os, time, json, sys
base = 'outputs/rtx5090_nlp_mini_ablation'
pa_values = [('0.5','0.5'), ('0.6','0.4'), ('0.7','0.3'),
              ('0.8','0.2'), ('0.9','0.1'), ('1.0','0.0')]
all_seeds = (42, 123, 456)

# Wait for Phase F (18 runs)
while True:
    missing = [f'pa{pa}_s{s}' for pa, pi in pa_values for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase F: still {len(missing)}/18 missing')
    sys.stdout.flush()
    time.sleep(60)

# Pick BEST_PA_F
best_ppl = float('inf'); best_pa = None; best_pi = None
for pa, pi in pa_values:
    ppls = [json.load(open(os.path.join(base, f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json')))['best_val_ppl']
            for s in all_seeds]
    m = sum(ppls) / 3
    if m < best_ppl:
        best_ppl = m; best_pa = pa; best_pi = pi
print(f'  BEST_PA_F = {best_pa} (Phase F mean PPL {best_ppl:.2f})')

# Wait for Phase G (9 runs: 3 wr × 3 seeds at BEST_PA_F)
while True:
    missing = [f'wr{wr}_s{s}' for wr in ('0.0','0.2','0.5') for s in all_seeds
               if not os.path.exists(os.path.join(base,
                   f'phaseG_wr{wr}_pa{best_pa}_s{s}', 'results.json'))]
    if not missing:
        break
    print(f'  Phase G: still {len(missing)}/9 missing')
    sys.stdout.flush()
    time.sleep(60)
print('  Phase F + G complete. Computing BEST_WR_G...')
EOF

# Compute BEST_PA + BEST_WR (wr=1.0 comes from Phase F at BEST_PA_F; wr<1 from Phase G)
read BEST_PA BEST_WR <<< $($PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'

best_ppl = float('inf'); best_pa = None; best_pi = None
for pa, pi in [('0.5','0.5'), ('0.6','0.4'), ('0.7','0.3'),
                ('0.8','0.2'), ('0.9','0.1'), ('1.0','0.0')]:
    ppls = [json.load(open(os.path.join(base, f'phaseF_pa{pa}_pi{pi}_s{s}', 'results.json')))['best_val_ppl']
            for s in (42, 123, 456)]
    m = sum(ppls) / 3
    if m < best_ppl:
        best_ppl = m; best_pa = pa; best_pi = pi

phaseG = {'1.0': best_ppl}  # wr=1.0 from Phase F
for wr in ('0.0', '0.2', '0.5'):
    ppls = [json.load(open(os.path.join(base, f'phaseG_wr{wr}_pa{best_pa}_s{s}', 'results.json')))['best_val_ppl']
            for s in (42, 123, 456)]
    phaseG[wr] = sum(ppls) / 3
best_wr = min(phaseG, key=phaseG.get)
print(f'{best_pa} {best_wr}')
EOF
)
if [ -z "$BEST_PA" ] || [ -z "$BEST_WR" ]; then
    echo "ERROR: could not determine BEST_PA/BEST_WR from Phases F+G"; exit 1
fi
BEST_PI=$($PYTHON -c "print(round(1.0 - $BEST_PA, 2))")
echo ""
echo "============================================================"
echo " NLP mini-ablation Phase H — SE ratio sweep (${#SEEDS[@]} seeds)"
echo " Fixed: pa=$BEST_PA pi=$BEST_PI (Phase F), wr=$BEST_WR (Phase F|G)"
echo " $(date)"
echo "============================================================"

COMMON_TRAINING="'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw', 'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000, 'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'"
COMMON_DATA="'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama', 'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100000000"
LM_SHARED="'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6, 'num_heads': 6, 'max_seq_len': 512"

run_se() {
    local X=$1 SEED=$2
    read BRANCH_FFN SE_DIM <<< $($PYTHON -c "X=$X; K=7; T=1536; print(round(T/(K+X)), round(T*X/(K+X)))")
    local ENAME="phaseH_se${X}x_pa${BEST_PA}_wr${BEST_WR}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    local TMPYAML="${ODIR}/_tmp.yaml"
    $PYTHON -c "
import yaml
cfg = {
    'model': {'type': 'multi_branch_transformer_lm', $LM_SHARED, 'num_branches': 7, 'ffn_dim_per_branch': $BRANCH_FFN, 'shared_expert_dim': $SE_DIM, 'dropout': 0.1},
    'algorithm': {'type': 'soft_specdrop', 'p_active': $BEST_PA, 'p_inactive': $BEST_PI, 'assignment': 'round_robin', 'warmup_ratio': $BEST_WR},
    'training': {$COMMON_TRAINING},
    'data': {$COMMON_DATA},
}
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $SEED
cfg['experiment_name'] = '$ENAME'
with open('$TMPYAML', 'w') as f:
    yaml.dump(cfg, f)
"
    echo "  Running $ENAME (branch_ffn=$BRANCH_FFN, se_dim=$SE_DIM) ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "$TMPYAML" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase H] seed=$SEED"
    for X in 0.5 1.0 4.0; do
        run_se "$X" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase H summary"
echo "============================================================"
BEST_PA_ENV="$BEST_PA" BEST_PI_ENV="$BEST_PI" BEST_WR_ENV="$BEST_WR" $PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
best_pa = os.environ['BEST_PA_ENV']; best_pi = os.environ['BEST_PI_ENV']; best_wr = os.environ['BEST_WR_ENV']
print(f"All runs use pa={best_pa} pi={best_pi} wr={best_wr} (from Phases F+G).")
print(f"{'SE ratio':>10}  {'mean PPL':>10}  {'std':>6}  n seeds  {'note':<40}")

# SE=0 reference: Phase F if best_wr=1.0, else Phase G
if best_wr == '1.0':
    paths = [os.path.join(base, f'phaseF_pa{best_pa}_pi{best_pi}_s{s}', 'results.json') for s in (42, 123, 456)]
    ref_note = '(ref: Phase F best_pa, wr=1.0)'
else:
    paths = [os.path.join(base, f'phaseG_wr{best_wr}_pa{best_pa}_s{s}', 'results.json') for s in (42, 123, 456)]
    ref_note = f'(ref: Phase G wr={best_wr})'
ppls = [json.load(open(p))['best_val_ppl'] for p in paths if os.path.exists(p)]
rows = []
if ppls:
    m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
    print(f"{'0x*':>10}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}  {ref_note}")

for X in ('0.5x', '1.0x', '4.0x'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseH_se{X[:-1]}_pa{best_pa}_wr{best_wr}_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5
        rows.append((X, m, std, len(ppls)))
        print(f"{X:>10}  {m:>10.2f}  {std:>6.2f}  {len(ppls):>7}")

complete = [r for r in rows if r[3] == 3]
if complete:
    best = min(complete, key=lambda r: r[1])
    print(f"\nBEST_SE_H (3-seed mean): {best[0]} → {best[1]:.2f} PPL")
    print(f"\nFINAL OPTIMUM for NLP 100M (CIFAR-defaults chain F→G→H):")
    print(f"  pa={best_pa}, wr={best_wr}, SE_ratio={best[0]}")
EOF
echo "Done: $(date)"
