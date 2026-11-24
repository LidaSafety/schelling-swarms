#!/bin/bash
# Pod I, one H200: Gemma 3 27B alone at N = 8, 16, 32, 64 and 128 (entry 21 of configs/experiments.yaml, benchmark
# branch mixed-n16), smallest first so a cut-off still leaves the smaller cells. Then mix16.sh
# (started with SKIP_GEMMA=1) waits for LANE_DONE and runs the N = 16 mixes.
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH      # ninja, for flashinfer's JIT build of the sampling kernels
export CUDA_HOME=/usr/local/cuda
cd /workspace/schelling-collusion-benchmark
SW="/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml"
wait_for() {   # wait_for <text in a log> <log>
  while ! grep -q "$1" "$2" 2>/dev/null; do sleep 15; done; echo "READY $1 $(date -u +%FT%TZ)"
}
echo "LANE_STARTED $(date -u +%FT%TZ)"
wait_for "ENV_OK" /workspace/logs/setup.log
wait_for "DOWNLOADING Qwen/Qwen3-8B" /workspace/logs/download.log               # Gemma is down
E="swarm_gemma_ladder_n8 swarm_gemma_ladder_n16 swarm_gemma_ladder_n32 swarm_gemma_ladder_n64 swarm_gemma_ladder_n128"
# shellcheck disable=SC2086
$SW --models gemma-3-27b --only $E     # one server for all five, in file order; results/.done marks each cell
for e in $E; do
  [ -f "results/.done/gemma-3-27b__$e" ] || { echo "LANE_FAILED gemma-3-27b $e $(date -u +%FT%TZ)"; exit 1; }
done
echo "LANE_DONE $(date -u +%FT%TZ)"
