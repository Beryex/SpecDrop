#!/bin/bash
###############################################################################
# RTX 5090 Script 3j: NLP mini-ablation — Phase J (pa sweep with NON-UNIFORM
# data-proportional branches at wr=1.0, SE=0).
#
# Parallel to Phase F but with heterogeneous per-branch ffn widths derived
# from the 100M cache's domain distribution:
#     ffn_dims[k] = round(1540 * frac_k) in domain_id order
# Total dense-equivalent budget = 1540 (matches uniform ffn=220 × K=7).
#
# Motivation (Phase I single-cell validation at pa=0.7):
#     uniform     pa=0.7 wr=1.0 SE=0 → 57.45 PPL
#     non-uniform pa=0.7 wr=1.0 SE=0 → 56.28 PPL   (−1.17, 6/6 domains improved)
# This is the first NLP config where ours beats the no_routing baseline (56.70).
# Phase J sweeps pa across the full range to locate BEST_PA_NU. The pa=0.7
# cell already exists (Phase I output dir) so only 5 new pa values × 3 seeds
# need running; pa=0.7 is auto-skipped.
#
# Output: outputs/rtx5090_nlp_mini_ablation/nonuniform_pa{pa}_wr1.0_s{seed}/
# (Uses the same dir naming as Phase I so pa=0.7 auto-skips the already-done runs.)
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase J — non-uniform pa sweep @ wr=1.0 SE=0"
echo " (data-proportional ffn_dims derived from 100M cache)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="nonuniform_pa${PA}_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import find_tokenize_cache, compute_proportional_ffn_dims

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] cache: {cache}')
ffn_dims, mapping = compute_proportional_ffn_dims(cache, total_ffn=1540, num_branches=7)
print(f'[cfg-gen] branch ↔ domain ↔ ffn (domain_id order, pa={PA}, wr=1.0, SE=0):')
for m in mapping:
    print(f"  b{m['branch']}: {m['domain_name']:<30} {m['frac']:>7.2%}  ffn={m['ffn_dim']}")
print(f'[cfg-gen] sum(ffn_dims)={sum(ffn_dims)}  max/min={max(ffn_dims)}/{min(ffn_dims)}')
assert mapping[0]['domain_name'] == 'RedPajamaCommonCrawl'
assert ffn_dims[0] == max(ffn_dims), "CC must get the largest branch"

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dims_per_branch': ffn_dims, 'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0},
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
    echo "[Phase J] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase J summary (non-uniform pa sweep @ wr=1.0 SE=0)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
print(f"{'pa':>5} {'pi':>5}  {'mean PPL':>10}  {'std':>6}  n")
rows = []
for pa in ['0.5','0.6','0.7','0.8','0.9','1.0']:
    pi = str(round(1 - float(pa), 2))
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'nonuniform_pa{pa}_wr1.0_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
complete = [r for r in rows if r[4] == 3]
if complete:
    best = min(complete, key=lambda r: r[2])
    print(f"\nBEST_PA_J (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
    print(f"Uniform Phase F BEST_PA=0.5 reference: 56.66 PPL")
    print(f"no_routing (pa=0.5 uniform) @ 100M:      ~56.70 PPL")
    print(f"no_routing+SE=1.0 (Phase D pa=0.5) @100M: 55.18 PPL")
EOF
echo "Done: $(date)"
