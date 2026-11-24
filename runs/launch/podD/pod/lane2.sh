#!/bin/bash
# One H200, one model at a time. Gates hold a block until its predictions are committed locally. Stops at the first
# block that does not finish, so a failure cannot cascade.
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH      # ninja, for flashinfer's JIT build of the sampling kernels
export CUDA_HOME=/usr/local/cuda
cd /workspace/schelling-collusion-benchmark
SW="/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml"
wait_gate() { while [ ! -f "$1" ]; do sleep 20; done; echo "GATE_OPEN $1 $(date -u +%FT%TZ)"; }
block() {   # block <label> <model> <experiment>...
  local label=$1 model=$2; shift 2
  $SW --models "$model" --only "$@"
  for e in "$@"; do
    if [ ! -f "results/.done/${model}__${e}" ]; then echo "LANE_FAILED $label $model $e $(date -u +%FT%TZ)"; exit 1; fi
  done
  echo "${label}_DONE $(date -u +%FT%TZ)"
}
echo "LANE_STARTED $(date -u +%FT%TZ)"
block GEMMA_SMALL gemma-3-27b wikigame_gemma_small wikigame_gemma_small_crowding
wait_gate /workspace/GATE_14B
block QWEN14 qwen3-14b wikigame_n128 wikigame_purity_s1 wikigame_purity_s24 wikigame_alone wikigame_ruleoff swarm_ruleoff
block QWEN4 qwen3-4b wikigame_alone
block QWEN8 qwen3-8b wikigame_ruleoff
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-14B /workspace/hf/hub/models--Qwen--Qwen3-4B /workspace/hf/hub/models--Qwen--Qwen3-8B
block QWEN32 qwen3-32b-bf16 swarm_ruleoff
wait_gate /workspace/GATE_GEMMA2
block GEMMA_LARGE gemma-3-27b wikigame_gemma_large
echo "LANE_DONE $(date -u +%FT%TZ)"
