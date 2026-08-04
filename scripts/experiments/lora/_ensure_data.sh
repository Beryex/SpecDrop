#!/bin/bash
# Ensure the SuperNI task JSONs are present (shallow clone of
# allenai/natural-instructions, ~400 MB one-time). The smoke script
# (scripts/experiments/smoke/lora.sh) has a hardened variant with retries
# and mirror fallback; this is the minimal guard for the training chains.
NI_DIR=${NI_DIR:-./data_cache/lora/natural-instructions}
if [ ! -d "$NI_DIR/tasks" ]; then
    echo "[lora] SuperNI data not found — cloning allenai/natural-instructions (~400 MB) ..."
    mkdir -p "$(dirname "$NI_DIR")"
    git clone --depth 1 https://github.com/allenai/natural-instructions.git "$NI_DIR" \
        || { echo "[lora] clone failed — see scripts/experiments/smoke/lora.sh for the retry/mirror variant"; exit 1; }
fi
