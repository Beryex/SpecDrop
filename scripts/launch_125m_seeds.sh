#!/bin/bash
# 125M scale-up seeds (paper App. E.10). Requires up to 4 GPUs.
#   bash scripts/launch_125m_seeds.sh
# Runs {ours, no_routing_se05} x seeds {42, 123, 456}; completed cells are
# skipped, so a fresh clone trains all six (~1 day each in parallel).
set -e
cd "$(dirname "$0")/.."
# Optionally export HF_HOME to point at a shared dataset/model cache.
PY=${PYTHON:-python}

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPU" -lt 1 ]; then
  echo "ERROR: no GPU found."; exit 1
fi

# One run per GPU; when more pending cells than GPUs, launch in waves.
i=0
for cfgname in ours_125m no_routing_se05_125m; do
  for seed in 42 123 456; do
    ODIR="./outputs/nlp_${cfgname%_125m}_125m/s${seed}"
    if [ -f "$ODIR/results.json" ]; then echo "skip $ODIR (done)"; i=$((i+1)); continue; fi
    mkdir -p "$ODIR"
    $PY -c "
import yaml
cfg = yaml.safe_load(open('configs/nlp/${cfgname}.yaml'))
cfg['output_dir'] = '$ODIR'
cfg['seed'] = $seed
cfg['experiment_name'] = '${cfgname}_s${seed}'
yaml.dump(cfg, open('/tmp/${cfgname}_s${seed}.yaml', 'w'))
"
    GPU=$((i % NGPU))
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY run_nlp.py \
      --config /tmp/${cfgname}_s${seed}.yaml --device cuda --no-wandb \
      > "$ODIR/train.log" 2>&1 < /dev/null &
    echo "GPU $GPU <- ${cfgname} s${seed} (pid $!)"
    i=$((i+1))
    if [ $((i % NGPU)) -eq 0 ]; then
      echo "-- wave full ($NGPU GPUs busy); waiting for it to finish --"
      wait
    fi
  done
done
wait
echo "All pending 125M cells finished. Results: outputs/nlp_*_125m/s*/results.json"
