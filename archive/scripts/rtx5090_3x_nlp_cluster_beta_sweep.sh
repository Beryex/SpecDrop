#!/bin/bash
###############################################################################
# RTX 5090 Script 3x: NLP mini-ablation — Phase W β sweep (cluster labels).
#
# Chains onto Phase W (rtx5090_3w): same root config (uniform + SE=1.0 +
# per-cat + wr=1.0 cosine + cluster labels) but extends the pa × β grid:
#     pa ∈ {0.5, 0.6, 0.7, 0.8}   (adds pa=0.8)
#     β  ∈ {1.0, 2.0, 4.0}        (adds β=2, β=4)
#
# Rationale. Phase V (β sweep with *domain* labels) found β>1 gives <0.1 PPL
# rescue — not significant. Phase W (β=1 with *cluster* labels) at pa=0.5/0.6
# shows tied-with-domain behavior. The open question: does β amplification
# work better on balanced semantic clusters than on imbalanced source domains?
# Cluster fracs are balanced (max/min=2.8×) vs domain (14×), so β's effect on
# gap_c = g·[(1-frac_c)/(1-1/M)]^β behaves differently.
#
# Skip logic:
#     - pa=0.5 × β>1:  bit-identical to β=1 because g=pa-pi=0 → gap_c=0 ∀c.
#     - pa ∈ {0.5,0.6,0.7} × β=1:  already covered by Phase W (reuse those dirs).
#     - Net runs: 7 new cells per seed × 3 seeds = 21 runs across 3 GPUs.
#
# Total wall: ~8h at ~70 min/run, 7 runs per GPU.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseW_cluster_pa{pa}_wr1.0_beta{β}_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

TRAIN_CLUSTER="./data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt"
VAL_CLUSTER="./data_cache/slimpajama/clusters_val_seq512_tok5000000_bge-large_k7.pt"

if [ ! -f "$TRAIN_CLUSTER" ] || [ ! -f "$VAL_CLUSTER" ]; then
    echo "ERROR: cluster caches missing (expected Phase W to have run)"
    echo "  train: $TRAIN_CLUSTER"
    echo "  val:   $VAL_CLUSTER"
    exit 1
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase W β sweep — cluster labels"
echo " (uniform + SE=1.0 + per-cat + wr=1.0 cosine, pa × β grid)"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_cell() {
    local SEED=$1 PA=$2 PI=$3 BETA=$4
    local ENAME="phaseW_cluster_pa${PA}_wr1.0_beta${BETA}_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI BETA=$BETA \
      TRAIN_CLUSTER=$TRAIN_CLUSTER VAL_CLUSTER=$VAL_CLUSTER \
      $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, load_cluster_labels)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
BETA = float(os.environ['BETA'])
TRAIN_CLUSTER = os.environ['TRAIN_CLUSTER']
VAL_CLUSTER = os.environ['VAL_CLUSTER']
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] train cache: {cache}  β={BETA}')
print(f'[cfg-gen] train cluster: {TRAIN_CLUSTER}')

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[cfg-gen] ffn/branch={ffn_per_branch}  SE_dim={se_dim}'
      f'  total={7*ffn_per_branch + se_dim}')

fracs = compute_category_fractions(
    cache, num_categories=7, cluster_label_path=TRAIN_CLUSTER)
cblob = load_cluster_labels(TRAIN_CLUSTER)
print(f'[cfg-gen] per-cluster fracs (β={BETA}):')
for c in range(7):
    print(f"  cluster_{c}                            frac={fracs[c]:>7.2%}")
assert abs(sum(fracs) - 1.0) < 1e-4

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': ffn_per_branch,
        'shared_expert_dim': se_dim,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': BETA,
                   'warmup_schedule': 'cosine'},
    'training': {'epochs': 10, 'batch_size': 64, 'lr': 3e-4, 'optimizer': 'adamw',
                  'weight_decay': 0.1, 'lr_schedule': 'cosine', 'warmup_steps': 1000,
                  'max_grad_norm': 1.0, '_compile_mode': 'reduce-overhead'},
    'data': {'dataset': 'slimpajama', 'data_dir': './data_cache/slimpajama',
              'num_workers': 4, 'max_seq_len': 512, 'max_train_tokens': 100_000_000,
              'cluster_label_path_train': TRAIN_CLUSTER,
              'cluster_label_path_val': VAL_CLUSTER},
    'output_dir': ODIR, 'seed': SEED, 'experiment_name': ENAME,
}
with open(os.path.join(ODIR, '_tmp.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
    echo "  Running $ENAME ... ($(date))"
    $PYTHON run_nlp.py --wandb --config "${ODIR}/_tmp.yaml" --device $DEVICE 2>&1 | tee "${ODIR}/${ENAME}.log"
    echo "  $ENAME finished: $(date)"
}

# Check whether a (pa, β=1.0) cell was already run by Phase W (different dir).
already_done_phaseW_beta1() {
    local PA=$1 SEED=$2
    local PATH_OLD="${OUTDIR_BASE}/phaseW_cluster_pa${PA}_wr1.0_s${SEED}/results.json"
    [ -f "$PATH_OLD" ]
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase W β sweep] seed=$SEED"
    for PA in 0.5 0.6 0.7 0.8; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        for BETA in 1.0 2.0 4.0; do
            # Skip pa=0.5 × β>1 — g=0 at pa=0.5 makes gap_c=0 for any β,
            # so these runs are mathematically bit-identical to pa=0.5 β=1.0.
            if [ "$PA" = "0.5" ] && [ "$BETA" != "1.0" ]; then
                echo "  skip pa=$PA β=$BETA (g=0 equivalent to β=1)"
                continue
            fi
            # Skip pa ∈ {0.5, 0.6, 0.7} × β=1 if Phase W (different dir scheme) already did it.
            if [ "$BETA" = "1.0" ] && already_done_phaseW_beta1 "$PA" "$SEED"; then
                echo "  skip pa=$PA β=1 (already run as Phase W)"
                continue
            fi
            run_cell "$SEED" "$PA" "$PI" "$BETA"
        done
    done
done

echo ""
echo "============================================================"
echo " Phase W β-sweep summary (pa × β, cluster labels)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
def _load(name):
    p = os.path.join(base, name, 'results.json')
    return json.load(open(p))['best_val_ppl'] if os.path.exists(p) else None

rows = {}
for pa in ('0.5', '0.6', '0.7', '0.8'):
    for beta in ('1.0', '2.0', '4.0'):
        ppls = []
        for s in (42, 123, 456):
            # Try β-explicit name first (new scheme)
            v = _load(f'phaseW_cluster_pa{pa}_wr1.0_beta{beta}_s{s}')
            # β=1 fallback to Phase W original name
            if v is None and beta == '1.0':
                v = _load(f'phaseW_cluster_pa{pa}_wr1.0_s{s}')
            if v is not None:
                ppls.append(v)
        if ppls:
            m = sum(ppls) / len(ppls)
            std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
            rows[(pa, beta)] = (m, std, len(ppls))

print(f"{'pa':>4} {'β':>5}  {'mean':>8} {'std':>6}  n")
for pa in ('0.5', '0.6', '0.7', '0.8'):
    for beta in ('1.0', '2.0', '4.0'):
        if (pa, beta) in rows:
            m, s, n = rows[(pa, beta)]
            print(f"{pa:>4} {beta:>5}  {m:>8.2f} {s:>6.3f}  {n}")
    print()

# BEST_PA check: for each β, find the argmin pa.
print("BEST_PA per β (3-seed means only):")
for beta in ('1.0', '2.0', '4.0'):
    candidates = {pa: rows[(pa, beta)][0]
                  for pa in ('0.5', '0.6', '0.7', '0.8')
                  if (pa, beta) in rows and rows[(pa, beta)][2] == 3}
    if candidates:
        best_pa = min(candidates, key=candidates.get)
        print(f"  β={beta}: pa={best_pa} → {candidates[best_pa]:.2f} PPL")
        # Compare to Phase V (domain-label) β-sweep reference at pa=0.6/0.7:
        # Phase V β=2 at pa=0.7 = 55.46 (domain); cluster equivalent here.

# Absolute floor reference
print(f"\nReference: Phase D pa=0.6 = 55.15 (mini-ablation absolute floor)")
EOF
echo "Done: $(date)"
