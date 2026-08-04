#!/bin/bash
###############################################################################
# RTX 5090 Script 3m: NLP mini-ablation — Phase M (per-category routing with
# UNIFORM branches, pa sweep at wr=1.0, SE=0).
#
# Purpose. Phase J/K/L showed non-uniform-branch runs beat no_routing by
# ~1.3 PPL, but BEST_PA stayed at 0.5, which is the routing-off degenerate
# point — the method's core routing mechanism is inert. Phase M tests
# whether per-category (p_a^c, p_i^c) routing, derived from data-proportional
# fracs, breaks the BEST_PA=0.5 degeneracy on uniform branches (where no
# other mechanism is at play).
#
# Per-category mapping (see algorithms/soft_specdrop.py for math):
#     gap_c = (p_a - p_i) * ((1 - frac_c) / (1 - 1/M))**β   (β=1 here)
#     p_a^c = S/K + gap_c * (K-1)/K
#     p_i^c = S/K - gap_c / K
# S is invariant → fixed-denominator merge + Theorem stack unchanged.
# Balanced fracs recover scalar (p_a, p_i) bit-identically.
#
# Search: pa ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, pi = 1 − pa
#   At pa=1.0 the small-frac categories' raw p_a^c would exceed 1 and
#   p_i^c would go negative. SoftSpecDrop clamps to [0, 1]; this is
#   S-invariant EXACTLY at pa=1.0 because S=1 and the clamped (1, 0)
#   row also sums to 1. Clamped categories snap to hard routing while
#   large categories stay at a soft weighting, producing an asymmetric
#   hard/soft hybrid that is still a legitimate datapoint to include.
# Fixed:  K=7, ffn=220 UNIFORM, no SE, wr=1.0, 100M tokens, 10 ep,
#         frac_per_category derived from 100M cache (CC 53.4%, ..., SE 3.8%)
# Compare to Phase F: same config without per-cat routing.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseM_pa{pa}_wr1.0_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase M — per-category routing @ uniform"
echo " branches, pa sweep @ wr=1.0 SE=0 (β=1.0)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseM_pa${PA}_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import find_tokenize_cache, compute_category_fractions, DOMAIN_NAMES

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] cache: {cache}')
fracs = compute_category_fractions(cache, num_categories=7)
print(f'[cfg-gen] per-category fractions (domain_id order, pa={PA}, wr=1.0, SE=0):')
for c, name in enumerate(DOMAIN_NAMES):
    print(f"  c{c}: {name:<30} {fracs[c]:>7.2%}")
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
# Sanity: CC (id=0) has the largest frac.
assert max(range(7), key=lambda i: fracs[i]) == 0, 'CC (id=0) should have max frac'

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': 220, 'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': 1.0},
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
    echo "[Phase M] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8 0.9 1.0; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase M summary (per-cat @ uniform branches, pa sweep)"
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
        p = os.path.join(base, f'phaseM_pa{pa}_wr1.0_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls); std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
        rows.append((pa, pi, m, std, len(ppls)))
        print(f"{pa:>5} {pi:>5}  {m:>10.2f}  {std:>6.2f}  {len(ppls)}")
complete = [r for r in rows if r[4] == 3]
if complete:
    best = min(complete, key=lambda r: r[2])
    print(f"\nBEST_PA_M (3-seed mean): pa={best[0]} pi={best[1]} → {best[2]:.2f} PPL")
    print(f"Reference — Phase F (uniform, scalar routing, wr=1.0 SE=0):")
    print(f"  pa=0.5 → 56.66, pa=0.7 → 57.45, pa=0.9 → 59.86, pa=1.0 → 64.60")
    print(f"  BEST_PA_F = 0.5 (routing-off degenerate point)")
    print(f"Hypothesis: if BEST_PA_M > 0.5, per-cat routing breaks the degeneracy.")
EOF
echo "Done: $(date)"
