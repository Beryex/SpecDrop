#!/bin/bash
###############################################################################
# RTX 5090 Script 3z: NLP mini-ablation — Phase Z (hardcoded k=3 clustering).
#
# Chains after Phase W (k=7) and Phase W β-sweep (3x). Retrains soft_specdrop
# at pa ∈ {0.5, 0.6} × 3 seeds with a 3-way clustering of BGE embeddings,
# keeping num_branches = num_categories = k = 3.
#
# Why k=3: silhouette scan over k ∈ [2, 50] gave argmax at k=3 (0.0314). All
# three internal validity metrics (silhouette / Calinski-Harabasz / Davies-
# Bouldin) were in the "no substantial structure" regime (Kaufman & Rousseeuw
# silhouette < 0.25) and did NOT converge on a single k — CH monotonically
# preferred k=2, DB argmin fell into singleton-cluster artifacts at k=47.
# BGE embeddings of SlimPajama chunks form a continuous manifold rather than
# discrete modes. k=3 is reported here as the silhouette-optimal choice to
# provide a counterpoint to Phase W's k=7; we expect within-noise PPL given
# that the silhouette gap (k=3 vs k=7) is only 0.006 (both 0.019-0.031).
#
# All cells keep the same root config:
#     uniform branches, SE_ratio=1.0, per-cat β=1.0, wr=1.0 cosine,
#     100M tokens × 10 ep, 30M transformer.
# The only change vs Phase W is (a) k (and thus num_branches) and (b) the
# cluster cache used for category labels.
#
# Per-branch FFN width is recomputed via split_ffn_budget_for_se to keep
# total params ≈ 1540 regardless of k (at k=3, ffn/branch=385, SE=385).
#
# Output: outputs/rtx5090_nlp_mini_ablation/phaseZ_cluster_k3_pa{pa}_wr1.0_s{seed}/
###############################################################################

PYTHON=${PYTHON:-python}
OUTDIR_BASE="./outputs/rtx5090_nlp_mini_ablation"
DEVICE="cuda"
if [ -n "$SEEDS_OVERRIDE" ]; then SEEDS=($SEEDS_OVERRIDE); else SEEDS=(42 123 456); fi

# Hardcoded — decision documented at top of file.
K_STAR=3

EMB_CACHE="./data_cache/slimpajama/embeddings_train_seq512_tok100000000_bge-large-en-v1.5.pt"

if [ ! -f "$EMB_CACHE" ]; then
    echo "ERROR: train embedding cache not found at $EMB_CACHE"
    echo "Phase W must have completed and saved embeddings."
    exit 1
fi

mkdir -p "$OUTDIR_BASE"

echo "============================================================"
echo " NLP mini-ablation Phase Z — k=${K_STAR} clustering (hardcoded)"
echo " $(date)"
echo "============================================================"

# ── Step 2: cluster at k* (reuses embedding cache) ───────────────────────────
TRAIN_CLUSTER="./data_cache/slimpajama/clusters_train_seq512_tok100000000_bge-large_k${K_STAR}.pt"
VAL_CLUSTER="./data_cache/slimpajama/clusters_val_seq512_tok5000000_bge-large_k${K_STAR}.pt"

if [ -f "$TRAIN_CLUSTER" ] && [ -f "$VAL_CLUSTER" ]; then
    echo "[phaseZ] cluster caches for k=$K_STAR already present, skipping."
else
    echo "[phaseZ] re-clustering at k=$K_STAR (reuses cached embeddings)..."
    $PYTHON - <<PYEOF
from data.slimpajama import find_tokenize_cache
from data.cluster_chunks import build_cluster_cache

train_cache = find_tokenize_cache(
    data_dir='./data_cache/slimpajama', split='train',
    max_tokens=100_000_000, max_seq_len=512)
val_cache = find_tokenize_cache(
    data_dir='./data_cache/slimpajama', split='val',
    max_tokens=5_000_000, max_seq_len=512)
build_cluster_cache(
    train_cache_path=train_cache,
    train_output_path='$TRAIN_CLUSTER',
    val_cache_path=val_cache,
    val_output_path='$VAL_CLUSTER',
    n_clusters=$K_STAR,
    embedder='BAAI/bge-large-en-v1.5',
    seed=42, batch_size=32, device='cuda',
    detok_batch_size=1024, skip_if_exists=True,
)
PYEOF
fi

# ── Step 3: train at pa ∈ {0.5, 0.6} × 3 seeds ───────────────────────────────
run_cell() {
    local SEED=$1 PA=$2 PI=$3 K=$4
    local ENAME="phaseZ_cluster_k${K}_pa${PA}_wr1.0_s${SEED}"
    local ODIR="${OUTDIR_BASE}/${ENAME}"
    if [ -f "$ODIR/results.json" ]; then
        echo "  $ENAME — DONE, skipping"; return
    fi
    mkdir -p "$ODIR"
    ODIR=$ODIR ENAME=$ENAME SEED=$SEED PA=$PA PI=$PI K=$K \
      TRAIN_CLUSTER=$TRAIN_CLUSTER VAL_CLUSTER=$VAL_CLUSTER \
      $PYTHON - <<'PYEOF'
import os, yaml
from data.slimpajama import (
    find_tokenize_cache, compute_category_fractions,
    split_ffn_budget_for_se, load_cluster_labels)

ODIR = os.environ['ODIR']; ENAME = os.environ['ENAME']
SEED = int(os.environ['SEED']); PA = float(os.environ['PA']); PI = float(os.environ['PI'])
K = int(os.environ['K'])
TRAIN_CLUSTER = os.environ['TRAIN_CLUSTER']
VAL_CLUSTER = os.environ['VAL_CLUSTER']
SE_RATIO = 1.0

cache = find_tokenize_cache(data_dir='./data_cache/slimpajama',
                             max_tokens=100_000_000, max_seq_len=512)
print(f'[cfg-gen] K={K}  pa={PA}  cluster={os.path.basename(TRAIN_CLUSTER)}')

total_branch_ffn, se_dim = split_ffn_budget_for_se(
    total_ffn_budget=1540, se_ratio=SE_RATIO, num_branches=K)
ffn_per_branch = total_branch_ffn // K
print(f'[cfg-gen] ffn/branch={ffn_per_branch}  SE_dim={se_dim}  '
      f'total={K*ffn_per_branch + se_dim}')

fracs = compute_category_fractions(
    cache, num_categories=K, cluster_label_path=TRAIN_CLUSTER)
assert abs(sum(fracs) - 1.0) < 1e-4, f'fracs sum={sum(fracs)}'
print(f'[cfg-gen] per-cluster fracs (K={K}):')
for c in range(K):
    print(f"  cluster_{c:>2}                         frac={fracs[c]:>7.2%}")

cfg = {
    'model': {
        'type': 'multi_branch_transformer_lm',
        'vocab_size': 50257, 'hidden_dim': 384, 'num_layers': 6,
        'num_heads': 6, 'num_branches': K,
        'ffn_dim_per_branch': ffn_per_branch,
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
    echo "[Phase Z] seed=$SEED  k=$K_STAR"
    for PA in 0.5 0.6; do
        PI=$($PYTHON -c "print(round(1.0 - $PA, 2))")
        run_cell "$SEED" "$PA" "$PI" "$K_STAR"
    done
done

echo ""
echo "============================================================"
echo " Phase Z summary — optimal-k=${K_STAR} vs k=7 (Phase W)"
echo "============================================================"
K_STAR=$K_STAR $PYTHON - <<'EOF'
import json, os
K = int(os.environ['K_STAR'])
base = 'outputs/rtx5090_nlp_mini_ablation'

def _load(name):
    p = os.path.join(base, name, 'results.json')
    return json.load(open(p))['best_val_ppl'] if os.path.exists(p) else None

print(f"{'config':>24}  {'mean':>8} {'std':>6}  n")
for pa in ('0.5', '0.6'):
    for config, pattern in [
        (f'k=7  pa={pa} β=1', f'phaseW_cluster_pa{pa}_wr1.0_s{{s}}'),
        (f'k={K} pa={pa} β=1', f'phaseZ_cluster_k{K}_pa{pa}_wr1.0_s{{s}}'),
    ]:
        ppls = [_load(pattern.format(s=s)) for s in (42, 123, 456)]
        ppls = [v for v in ppls if v is not None]
        if ppls:
            m = sum(ppls)/len(ppls)
            std = (sum((x-m)**2 for x in ppls)/len(ppls))**0.5 if len(ppls)>1 else 0.0
            print(f"  {config:>22}  {m:>8.2f} {std:>6.3f}  {len(ppls)}")
    print()

print("Reference anchors:")
print("  Phase D pa=0.6 (scalar, domain) = 55.15  ← mini-ablation absolute floor")
print("  Phase P pa=0.6 (per-cat, domain, k=7) = 55.17")
print("  Phase W pa=0.5 (per-cat, cluster, k=7) = 55.11  ← current best")
EOF
echo "Done: $(date)"
