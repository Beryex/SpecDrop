#!/bin/bash
###############################################################################
# Batch-size sweep for Llama-3.2-1B LoRA. Holds effective_batch = 128 fixed.
# Runs PROFILE_STEPS=20 at each bs so total wall time is ~5-10 min.
#
# Usage (on 5090 with LoRA code checked out):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/profile_bs_sweep.sh
#
# Outputs:
#   outputs/bs_sweep_bs{8,16,32}.log — full per-bs log (tail for PROFILE block)
#   stdout: PROFILE + error summary per bs
#
# Exits cleanly at 20 steps via trainer's sys.exit(0); no training happens.
###############################################################################
set -eu

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
BASE_YAML=configs/lora/ours.yaml
TMP_YAML=$(mktemp -t ours_bs_sweep_XXXX.yaml)
mkdir -p outputs
trap 'rm -f "$TMP_YAML"' EXIT

# Test battery. bs × accum = 128 effective batch in every case.
declare -a SWEEP=("8 16" "16 8" "32 4")

echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│ LoRA BS sweep (Llama-3.2-1B, SuperNI, seq=1024, grad_ckpt=on)    │"
echo "│ Effective batch fixed at 128. Watching VRAM via nvidia-smi.      │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""

for SPEC in "${SWEEP[@]}"; do
    BS=$(echo $SPEC | cut -d' ' -f1)
    ACCUM=$(echo $SPEC | cut -d' ' -f2)
    EFFECTIVE=$((BS * ACCUM))
    LOG=outputs/bs_sweep_bs${BS}.log

    echo "════════════════════════════════════════════════════════════════"
    echo " BS=$BS  ACCUM=$ACCUM  → effective_batch=$EFFECTIVE"
    echo " full log: $LOG"
    echo "════════════════════════════════════════════════════════════════"

    # Generate a config with overridden bs/accum.
    $PYTHON <<PYEOF
import yaml
with open("$BASE_YAML") as f: cfg = yaml.safe_load(f)
cfg['training']['batch_size_per_device'] = $BS
cfg['training']['grad_accum_steps'] = $ACCUM
with open("$TMP_YAML", 'w') as f: yaml.safe_dump(cfg, f)
PYEOF

    # Run profile with ALL output to per-bs log.
    # Tolerate non-zero exit so the sweep continues past OOM.
    set +e
    PROFILE_STEPS=20 $PYTHON run_lora.py --config "$TMP_YAML" > "$LOG" 2>&1
    RC=$?
    set -e

    # Print a terse summary on stdout: the PROFILE block + any OOM lines.
    # The trainer prints a framed `[PROFILE] First 20 micro-steps` table,
    # followed by per-phase stats and a TOTAL/share section.
    if [ $RC -ne 0 ]; then
        echo "  [!] python exited $RC at bs=$BS (likely OOM). Tail of $LOG:"
        tail -20 "$LOG" | sed 's/^/    /'
    else
        # Print the PROFILE report block (from "[PROFILE]" line to "sys.exit(0)")
        awk '/\[PROFILE\] First/,/sys\.exit\(0\)/' "$LOG" | sed 's/^/    /'
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo " BS sweep complete. Summary comparison:"
for SPEC in "${SWEEP[@]}"; do
    BS=$(echo $SPEC | cut -d' ' -f1)
    LOG=outputs/bs_sweep_bs${BS}.log
    ACCUM=$(echo $SPEC | cut -d' ' -f2)
    TOTAL=$(grep "TOTAL:" "$LOG" 2>/dev/null | head -1 | sed 's/.*= /  /' || echo "  (no TOTAL — likely OOM)")
    printf "  bs=%-3s accum=%-3s → %s\n" "$BS" "$ACCUM" "$TOTAL"
done
echo ""
echo " Pick bs with:"
echo "   1. No OOM (check log tails above)"
echo "   2. Lowest per-step avg × accum = best opt-step wall time"
echo "   3. VRAM peak (nvidia-smi) with comfortable headroom (<28 GB)"
echo "════════════════════════════════════════════════════════════════"
