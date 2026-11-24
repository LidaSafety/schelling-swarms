#!/bin/bash
# Pod H, one H200, one model at a time. Every forecast for these runs was committed before this lane started
# (research repo, analysis/occupancy/predictions_pod5.md). Stops at the first block that does not finish.
# Within a model: the N = 8 control first, then the plant (N = 16), then N = 64, then N = 128, the longest.
# RUN_Q8=1 adds the optional first stage of 4 answers per copy (swarm_q8_small) to the 4B, 14B and 32B blocks.
set -u
RUN_Q8=${RUN_Q8:-0}
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH      # ninja, for flashinfer's JIT build of the sampling kernels
export CUDA_HOME=/usr/local/cuda
cd /workspace/schelling-collusion-benchmark
SW="/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml"
Q8=""; [ "$RUN_Q8" = 1 ] && Q8="swarm_q8_small"
wait_for() {   # wait_for <text in a log> <log>
  while ! grep -q "$1" "$2" 2>/dev/null; do sleep 15; done; echo "READY $1 $(date -u +%FT%TZ)"
}
block() {   # block <label> <model> <experiment>...
  local label=$1 model=$2; shift 2
  $SW --models "$model" --only "$@"
  for e in "$@"; do
    if [ ! -f "results/.done/${model}__${e}" ]; then echo "LANE_FAILED $label $model $e $(date -u +%FT%TZ)"; exit 1; fi
  done
  echo "${label}_DONE $(date -u +%FT%TZ)"
}
echo "LANE_STARTED $(date -u +%FT%TZ) RUN_Q8=$RUN_Q8"
wait_for "ENV_OK" /workspace/logs/setup.log
wait_for "DOWNLOADING Qwen/Qwen3-8B" /workspace/logs/download.log               # Qwen3-4B is down
block QWEN4 qwen3-4b swarm_scale_small swarm_plant swarm_scale_n64 swarm_scale_n128 $Q8
wait_for "DOWNLOADING Qwen/Qwen3-14B" /workspace/logs/download.log              # Qwen3-8B is down
block QWEN8 qwen3-8b swarm_scale_small swarm_scale1_small swarm_plant swarm_scale1_n64 swarm_scale_n128 swarm_scale1_n128
wait_for "DOWNLOADING Qwen/Qwen3-32B" /workspace/logs/download.log              # Qwen3-14B is down
block QWEN14 qwen3-14b swarm_scale_small swarm_plant swarm_scale_n64 swarm_scale_n128 $Q8
wait_for "DOWNLOAD_OK" /workspace/logs/download.log
block QWEN32 qwen3-32b-bf16 swarm_scale_small swarm_scale_n64 swarm_scale_n128 $Q8
echo "LANE_DONE $(date -u +%FT%TZ)"
