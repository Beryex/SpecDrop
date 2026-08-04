#!/bin/bash
###############################################################################
# RTX 5090 Script 3w: NLP mini-ablation — Phase W (semantic-cluster labels
# replacing SlimPajama source-level domain_ids at Phase P root).
#
# Motivation. Phase P fixed the "Phase D-equivalent with per-cat" config
# (uniform + SE=1.0 + per-cat + wr=1.0 cosine, β=1.0) and found pa=0.5/0.6
# tied at the empirical minimum 55.17 PPL. The per-cat amplification term
# (1 - frac_c)^β at β=1 was inert because SlimPajama's 7 source-level
# domain tags are noisy signals of semantic content (CC is a 53% bag of
# everything; Wikipedia contains ArXiv-style articles; etc.). Phase W tests
# the hypothesis that replacing the source-tag category with a SEMANTIC
# cluster label — produced offline by embedding each 512-token chunk with
# BGE-large-en-v1.5 and KMeans-clustering in embedding space — makes the
# per-category routing signal informative, potentially shifting BEST_PA
# above 0.5 or lowering the plateau below Phase D's 55.15 floor.
#
# All other config = Phase P exactly:
#   uniform ffn_dim=192, SE=192 (SE_ratio=1.0), per-cat β=1.0,
#   wr=1.0 cosine warmup, 100M tokens × 10 epochs, 30M transformer.
# ONLY difference: `frac_per_category` and the trainer's per-sample category
# labels come from cluster_ids, not domain_ids.
#
# Cluster cache produced offline by _run_gpu0.sh via
#   `python -m data.cluster_chunks ...` (see _run_gpu0.sh header).
# _run_gpu1/2.sh poll for the cache before starting.
#
# Search: pa ∈ {0.5, 0.6, 0.7}, pi = 1 − pa, wr=1.0 fixed, SE=1.0 fixed,
#         uniform branches, per-cat β=1.0.
# 3 pa × 3 seeds = 9 runs, ~2.5h wall-clock on 3 GPUs.
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseW_cluster_pa{pa}_wr1.0_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# Cluster cache paths (must match _run_gpu0.sh's clustering step).
TRAIN_CLUSTER="./data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k7.pt"
VAL_CLUSTER="./data_cache/slimpajama/clusters_val_seq512_tok5000000_bge-large_k7.pt"

if [ ! -f "$TRAIN_CLUSTER" ]; then
    echo "ERROR: train cluster cache missing at $TRAIN_CLUSTER"
    echo "Run _run_gpu0.sh (which produces it) first."
    exit 1
fi
if [ ! -f "$VAL_CLUSTER" ]; then
    echo "ERROR: val cluster cache missing at $VAL_CLUSTER"
    exit 1
fi

mkdir -p "$OUTDIR_BASE"
echo "============================================================"
echo " NLP mini-ablation Phase W — cluster labels @ Phase P root"
echo " (uniform + SE=1.0 + per-cat β=1.0 + wr=1.0 cosine, pa sweep)"
echo " Train cluster: $TRAIN_CLUSTER"
echo " Val cluster:   $VAL_CLUSTER"
echo " ${#SEEDS[@]} seed(s), $(date)"
echo "============================================================"

run_pa() {
    local PA=$1 PI=$2 SEED=$3
    local ENAME="phaseW_cluster_pa${PA}_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI \
      TRAIN_CLUSTER=$TRAIN_CLUSTER VAL_CLUSTER=$VAL_CLUSTER \
      $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, load_cluster_labels)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
TRAIN_CLUSTER = os.environ['TRAIN_CLUSTER']
VAL_CLUSTER = os.environ['VAL_CLUSTER']
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] train cache: {cache}')
print(f'[cfg-gen] train cluster labels: {TRAIN_CLUSTER}')
print(f'[cfg-gen] val cluster labels:   {VAL_CLUSTER}')

# Same budget split as Phase P.
total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=7)
ffn_per_branch = total_branch_ffn // 7
print(f'[cfg-gen] SE_ratio={SE_RATIO}  total_branch_ffn={total_branch_ffn}  SE_dim={se_dim}')
print(f'[cfg-gen] ffn_per_branch (uniform) = {ffn_per_branch}')
print(f'[cfg-gen] total params budget: {7*ffn_per_branch + se_dim} (target 1540)')

# Fractions come from cluster_ids, not SlimPajama source domain_ids.
fracs = compute_category_fractions(
    cache, num_categories=7, cluster_label_path=TRAIN_CLUSTER)
cblob = load_cluster_labels(TRAIN_CLUSTER)
print(f'[cfg-gen] per-cluster fractions (embedder={cblob.get("embedder")}, '
      f'k={cblob.get("n_clusters")}, seed={cblob.get("seed")}):')
for c in range(7):
    print(f"  cluster_{c:>2}                         {fracs[c]:>7.2%}")
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': 7,
        'ffn_dim_per_branch': ffn_per_branch,   # scalar → uniform
        'shared_expert_dim': se_dim,
        'max_seq_len': 512, 'dropout': 0.1,
    },
    'algorithm': {'type': 'soft_specdrop', 'p_active': PA, 'p_inactive': PI,
                   'assignment': 'round_robin', 'warmup_ratio': 1.0,
                   'frac_per_category': fracs, 'amplification_beta': 1.0,
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

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "[Phase W] seed=$SEED"
    for PA in 0.5 0.6 0.7; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_pa "$PA" "$PI" "$SEED"
    done
done

echo ""
echo "============================================================"
echo " Phase W summary — cluster labels vs Phase P (domain_ids)"
echo "============================================================"
$PYTHON - <<'EOF'
import json, os
base = 'outputs/rtx5090_nlp_mini_ablation'
# Phase P 3-seed means (from 2026-04-19 session) for Δ reference.
phaseP = {'0.5': 55.17, '0.6': 55.17, '0.7': 55.56}
print(f"{'pa':>4}  {'W mean':>8}  {'W std':>6}  {'P mean':>8}  {'Δ':>7}  n")
rows = {}
for pa in ('0.5', '0.6', '0.7'):
    ppls = []
    for s in (42, 123, 456):
        p = os.path.join(base, f'phaseW_cluster_pa{pa}_wr1.0_s{s}', 'results.json')
        if os.path.exists(p):
            ppls.append(json.load(open(p))['best_val_ppl'])
    if ppls:
        m = sum(ppls)/len(ppls)
        std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
        ref = phaseP[pa]
        rows[pa] = m
        print(f"{pa:>4}  {m:>8.2f}  {std:>6.2f}  {ref:>8.2f}  {m-ref:>+7.2f}  {len(ppls)}")

if rows and all(s == 3 for s in [len([1 for s in (42,123,456) if os.path.exists(
        os.path.join(base, f'phaseW_cluster_pa{pa}_wr1.0_s{s}', 'results.json'))])
        for pa in rows]):
    best = min(rows, key=rows.get)
    print(f"\nBEST_PA_W (3-seed mean): pa={best} → {rows[best]:.2f} PPL")
    phaseD = 55.15
    print(f"Reference (100M tokens):")
    print(f"  Phase D (uniform + SE=1.0, scalar, no per-cat)   BEST_PA=0.5/0.6 → 55.15")
    print(f"  Phase P (uniform + SE=1.0, per-cat, domain_ids)  BEST_PA=0.5/0.6 → 55.17")
    print(f"  Phase W (uniform + SE=1.0, per-cat, cluster_ids) above")
    if rows[best] < phaseD:
        print(f"\n✓ Phase W beats Phase D anchor by {phaseD - rows[best]:.2f} PPL "
              f"— semantic clusters unlock per-cat routing benefit.")
    else:
        print(f"\n✗ Phase W does not beat Phase D anchor (55.15); "
              f"semantic clusters alone insufficient at 30M/100M scale.")
EOF
echo "Done: $(date)"
