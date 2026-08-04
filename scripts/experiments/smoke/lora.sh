#!/bin/bash
###############################################################################
# RTX 5090 Script 0d: LoRA track — full setup + smoke test (one-shot).
#
# Designed to be the SINGLE script you run once on a freshly pulled 5090 box
# to get the entire LoRA pipeline ready. It:
#   (1) verifies HuggingFace login + gated-model access,
#   (2) pre-downloads Llama-3.2-1B base + tokenizer (~6 GB, one-time),
#   (3) shallow-clones allenai/natural-instructions (~400 MB, one-time),
#   (4) builds the K=20 Wang 2022 Domains mapping,
#   (5) pre-tokenizes + caches 3 splits to disk (train-full, train-20%, test)
#       so ablation / main-table runs never re-tokenize,
#   (6) runs a 100-example smoke through ALL 6 method configs on GPU 0 to
#       verify the full training pipeline is healthy end-to-end.
#
# After a clean pass of this script, the LoRA ablation / main-table chains
# (scripts/experiments/lora/) run entirely from the cached tensors — no
# re-tokenize time in any subsequent run.
#
# Runtime on RTX 5090:
#   (1)-(2) HF auth + model dl:          ~ 3-8 min (gated, 6 GB)
#   (3) SuperNI clone:                   ~ 1 min
#   (4) domain map:                      ~ 5 sec
#   (5) tokenize train-full/20%/test:    ~ 5-10 min combined (one-time)
#   (6) 6-method smoke run:              ~ 15-20 min (100 examples × 6 methods)
#   ────────────────────────────────────────────────────
#   Total:                                ~ 30-40 min one-time
#
# Env vars:
#   HF_TOKEN               (required if `huggingface-cli login` not already
#                           run on this box; will be passed to transformers)
#   BASE_MODEL             (default: meta-llama/Llama-3.2-1B)
#   DEVICE                 (default: cuda)
#   SKIP_SMOKE             (default: 0; set to 1 to skip step 6 and only
#                           prep caches — useful for pre-warming before a
#                           multi-GPU launch)
###############################################################################

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-1B}
NI_REPO=https://github.com/allenai/natural-instructions.git
NI_DIR=./data_cache/lora/natural-instructions
CACHE_DIR=./data_cache/lora
OUTDIR=./outputs/smoke_lora
SKIP_SMOKE=${SKIP_SMOKE:-0}

mkdir -p "$OUTDIR" "$CACHE_DIR" ./data_cache/lora

echo "============================================================"
echo " LoRA track: one-shot setup + smoke test"
echo " base=$BASE_MODEL  device=$DEVICE  cache=$CACHE_DIR"
echo " $(date)"
echo "============================================================"

# ── (1) HuggingFace auth check (Llama-3.2-1B is gated) ─────────────────────
# Uses the huggingface_hub Python SDK directly — this is the canonical path
# (shared by both `huggingface-cli` and the newer `hf` CLI, and identical to
# what transformers.AutoModelForCausalLM.from_pretrained uses internally).
# So this check works whichever CLI you logged in with.
echo ""
echo "[1/6] HuggingFace auth check"
HF_USER=$($PYTHON -c "
try:
    from huggingface_hub import whoami
    info = whoami()
    print(info.get('name') or info.get('fullname') or '')
except Exception:
    pass
" 2>/dev/null)

if [ -z "$HF_USER" ] && [ -n "$HF_TOKEN" ]; then
    echo "  Not logged in; using HF_TOKEN from env..."
    $PYTHON -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>&1 | tail -3
    HF_USER=$($PYTHON -c "
from huggingface_hub import whoami
try:
    info = whoami()
    print(info.get('name') or info.get('fullname') or '')
except Exception:
    pass
" 2>/dev/null)
fi

if [ -z "$HF_USER" ]; then
    echo "  ERROR: not logged in to HuggingFace Hub AND no HF_TOKEN env var set."
    echo "  Required because $BASE_MODEL is a gated model. Options:"
    echo "    1)  huggingface-cli login    (legacy CLI)"
    echo "    2)  hf auth login            (new CLI)"
    echo "    3)  export HF_TOKEN=hf_xxx   (env var, good for non-interactive)"
    exit 1
fi
echo "  HF auth OK"

# ── (2) Pre-download Llama-3.2-1B base (cached by HF hub across all runs) ───
echo ""
echo "[2/6] Pre-download $BASE_MODEL base + tokenizer"
$PYTHON - <<PYEOF
import os, sys
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
name = "$BASE_MODEL"
print(f"  loading tokenizer {name}...")
tok = AutoTokenizer.from_pretrained(name)
print(f"  loading model {name} (bf16) — first run downloads ~6 GB...")
# HF transformers 4.48 renamed torch_dtype → dtype (deprecation warning).
tf_ver = tuple(int(x) for x in transformers.__version__.split('.')[:2])
dtype_kw = 'dtype' if tf_ver >= (4, 48) else 'torch_dtype'
try:
    m = AutoModelForCausalLM.from_pretrained(name, **{dtype_kw: torch.bfloat16})
except Exception as e:
    print(f"  ERROR loading model: {e}")
    print("  Check: HF TOS accepted for $BASE_MODEL?  Correct HF_TOKEN scope?")
    sys.exit(1)
n = sum(p.numel() for p in m.parameters())
print(f"  {name} OK — {n/1e9:.2f} B params")
PYEOF

# ── (3) SuperNI clone (shallow, ~400 MB — retry + mirror fallback) ──────────
echo ""
echo "[3/6] SuperNI dataset"
if [ -d "$NI_DIR/tasks" ] && [ -f "$NI_DIR/splits/default/train_tasks.txt" ]; then
    N_TASKS=$(ls "$NI_DIR/tasks" | wc -l | tr -d ' ')
    echo "  already cloned → $NI_DIR ($N_TASKS task JSONs)"
else
    # Clean any partial clone from a prior failed attempt.
    if [ -d "$NI_DIR" ]; then
        echo "  cleaning partial/failed clone at $NI_DIR ..."
        rm -rf "$NI_DIR"
    fi
    mkdir -p "$(dirname "$NI_DIR")"
    # Boost git buffers so large sideband packets survive flaky GitHub
    # links (the "curl 18 transfer closed" error is the default 1-MB
    # postBuffer giving up on a ~400 MB fetch).
    GITBUF=(-c http.postBuffer=1048576000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999)

    # Try direct URL twice, then a ghproxy mirror (helps on restricted networks).
    URLS=(
        "$NI_REPO"
        "$NI_REPO"
        "https://ghproxy.com/$NI_REPO"
    )
    CLONED=0
    for i in "${!URLS[@]}"; do
        URL="${URLS[$i]}"
        echo "  attempt $((i+1))/3: git clone --depth 1 $URL"
        if git "${GITBUF[@]}" clone --depth 1 "$URL" "$NI_DIR" 2>&1 | tail -5; then
            if [ -f "$NI_DIR/splits/default/train_tasks.txt" ]; then
                CLONED=1
                break
            fi
        fi
        [ -d "$NI_DIR" ] && rm -rf "$NI_DIR"
        [ $i -lt 2 ] && { echo "  retrying in 5s..."; sleep 5; }
    done

    if [ "$CLONED" != "1" ]; then
        cat <<EOF
  ERROR: SuperNI clone failed after 3 retries.
  Options to recover:
    1) Retry this script — transient GitHub flakiness.
    2) Use a proxy / set HTTP(S)_PROXY then retry.
    3) Local-machine workaround (reliable):
         (on your laptop)  git clone --depth 1 $NI_REPO /tmp/ni
         (on your laptop)  rsync -avz -e "ssh -p <port>" /tmp/ni/ \\
                               <user>@<host>:<repo-root>/$NI_DIR/
EOF
        exit 1
    fi

    N_TASKS=$(ls "$NI_DIR/tasks" | wc -l | tr -d ' ')
    echo "  cloned $N_TASKS task JSONs"
fi

# ── (4) Build K=20 domain map ──────────────────────────────────────────────
echo ""
echo "[4/6] K=20 Wang 2022 Domains mapping"
$PYTHON - <<PYEOF
from data.superni_domain_map import build_or_load_domain_map
m = build_or_load_domain_map(
    tasks_dir="$NI_DIR/tasks",
    splits_dir="$NI_DIR/splits/default",
    cache_dir="$CACHE_DIR", K=20)
print(f"  K={m['K']}  train tasks covered: {len(m['task_to_cluster'])}")
print(f"  top-9 domains: {[d for d,_ in m['top_domains']]}")
print(f"  misc id: {m['misc_id']}")
PYEOF

# ── (5) Pre-tokenize + cache train (full + 20%) + test splits ──────────────
echo ""
echo "[5/6] Pre-tokenize SuperNI → local caches (one-time, 5-10 min)"
$PYTHON - <<PYEOF
import os, time
from transformers import AutoTokenizer
from data.superni_domain_map import build_or_load_domain_map
from data.natural_instructions import prebuild_all_caches

t0 = time.time()
tok = AutoTokenizer.from_pretrained("$BASE_MODEL")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
m = build_or_load_domain_map(
    tasks_dir="$NI_DIR/tasks",
    splits_dir="$NI_DIR/splits/default",
    cache_dir="$CACHE_DIR", K=20)

paths = prebuild_all_caches(
    data_root="$NI_DIR",
    tokenizer=tok, mapping=m,
    cache_dir="$CACHE_DIR",
    max_seq_len=1024,
    instances_per_task=100,
    seed=42,
    subset_fracs_train=(1.0, 0.2),   # full + 20% ablation subset
    num_pos_examples=2,
    num_neg_examples=0,
    verbose=True)

elapsed = time.time() - t0
print(f"\n  wrote {len(paths)} caches in {elapsed/60:.1f} min:")
for k, p in paths.items():
    sz = os.path.getsize(p) / 1e6
    print(f"    {k:20s} → {p}  ({sz:.1f} MB)")
PYEOF

# ── (6) Smoke test: 6 methods × 100 examples × 1 epoch ──────────────────────
if [ "$SKIP_SMOKE" = "1" ]; then
    echo ""
    echo "[6/6] Smoke test SKIPPED (SKIP_SMOKE=1 — caches are ready)."
    echo "Done: $(date)"
    exit 0
fi

echo ""
echo "[6/6] 6-method smoke run (single GPU, ~100 examples, 1 epoch)"
echo "      (uses the full 20%-subset cache you just built; same data as 8a/8b/8c)"

smoke_config() {
    local BASE_CFG=$1 OUT=$2
    # Use quoted heredoc + env vars to avoid bash substitution ambiguity.
    # Previous unquoted-heredoc version silently failed to apply the smoke
    # knobs on some shells, leaving subset_frac_train at the config default
    # (1.0 or 0.2) — smoke ended up training on the full 20% cache and
    # taking ~45 min/method instead of ~1 min.
    BASE_CFG="$BASE_CFG" OUT="$OUT" $PYTHON - <<'PYEOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ['BASE_CFG']))
cfg['training']['epochs'] = 1
cfg['training']['log_interval'] = 5
cfg['training']['eval_interval_epochs'] = 1
cfg['training']['run_rouge_eval'] = False   # generation eval is slow; skip in smoke
# Hard-cap at 20 training steps — the smoke only needs to verify pipeline,
# data wiring, gradient flow, and shape compat. 20 steps × ~0.7 s/step = 15 s
# training + ~60 s eval (119 test examples @ batch 2) + ~30 s HF load =
# ~2 min/method × 6 methods ≈ 12 min total smoke.
cfg['training']['max_steps'] = 20
cfg['training']['batch_size_per_device'] = 2
cfg['training']['grad_accum_steps'] = 1
# Subset knobs still small so DataLoader spin-up is fast + fits in RAM.
cfg['data']['subset_frac_train'] = 1.0
cfg['data']['instances_per_task_train'] = 2   # 2 × 756 ≈ 1500 examples
cfg['data']['instances_per_task_eval'] = 1    # 119 examples (eval skipped anyway)
cfg['data']['num_workers'] = 0
cfg['output_dir'] = os.environ['OUT']
cfg['seed'] = 42
print(f"  smoke cfg: frac={cfg['data']['subset_frac_train']} "
      f"ipt={cfg['data']['instances_per_task_train']} "
      f"batch={cfg['training']['batch_size_per_device']} "
      f"accum={cfg['training']['grad_accum_steps']}")
with open(os.path.join(os.environ['OUT'], '_smoke.yaml'), 'w') as f:
    yaml.dump(cfg, f)
PYEOF
}

for METHOD in single_lora_r16 loramoe hydra_lora mocle mb_lora_no_routing ours; do
    OUT="$OUTDIR/$METHOD"
    if [ -f "$OUT/results.json" ]; then
        echo "  $METHOD — smoke already done, skipping"
        continue
    fi
    mkdir -p "$OUT"
    echo "  [smoke] $METHOD"
    smoke_config "configs/lora/$METHOD.yaml" "$OUT"
    $PYTHON run_lora.py --config "$OUT/_smoke.yaml" --device $DEVICE 2>&1 \
        | tee "$OUT/smoke.log"
    echo "    $METHOD finished"
done

echo ""
echo "============================================================"
echo " Smoke results"
echo "============================================================"
FAILED=0
for METHOD in single_lora_r16 loramoe hydra_lora mocle mb_lora_no_routing ours; do
    R="$OUTDIR/$METHOD/results.json"
    if [ -f "$R" ]; then
        SUMMARY=$($PYTHON -c "
import json
d = json.load(open('$R'))
parts = [f\"eval_loss={d.get('best_eval_loss', float('nan')):.4f}\"]
if d.get('eval_rouge_l') is not None:
    parts.append(f\"ROUGE-L={d['eval_rouge_l']:.4f}\")
    parts.append(f\"EM={d.get('eval_exact_match', 0.0):.4f}\")
    parts.append(f\"best_epoch={d.get('best_epoch', '?')}\")
print(' | '.join(parts))
")
        echo "  $METHOD: $SUMMARY"
    else
        echo "  $METHOD: FAILED (no results.json)"
        FAILED=1
    fi
done
echo ""
if [ "$FAILED" = "1" ]; then
    echo "Some methods FAILED — check $OUTDIR/<method>/smoke.log before firing"
    echo "the multi-GPU ablation chain."
else
    echo "All 6 methods passed. You can now launch the full pipeline:"
    echo "    bash reproduce.sh lora                            # single-GPU sequential (~290 h)"
    echo "    # multi-GPU per-seed parallel (3 GPUs, ~95 h):"
    echo "    CUDA_VISIBLE_DEVICES=0 SEEDS_OVERRIDE=42  bash scripts/experiments/lora/main_table.sh &"
    echo "    CUDA_VISIBLE_DEVICES=1 SEEDS_OVERRIDE=123 bash scripts/experiments/lora/main_table.sh &"
    echo "    CUDA_VISIBLE_DEVICES=2 SEEDS_OVERRIDE=456 bash scripts/experiments/lora/main_table.sh &"
    echo "    wait"
fi
echo "Done: $(date)"
