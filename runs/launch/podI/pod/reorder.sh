#!/bin/bash
# Pod I, at the user's request (18 Nov, 19:40 UTC): the N = 16 mixes before Gemma at N = 128. Waits for the lane to
# mark N = 64 done, then stops the lane's sweep, its N = 128 run and its vLLM server. lane.sh then logs LANE_FAILED for
# N = 128 (expected), which releases mix16.sh. After the mixes, Gemma at N = 128 runs alone. Log: /workspace/logs/reorder.log
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/schelling-collusion-benchmark
echo "REORDER_STARTED $(date -u +%FT%TZ)"
while [ ! -f results/.done/gemma-3-27b__swarm_gemma_ladder_n64 ]; do sleep 5; done
echo "N64_DONE_SEEN $(date -u +%FT%TZ)"
pkill -TERM -f "scripts/sweep.py --models-file configs/models_h200.yaml --models gemma-3-27b"
sleep 2
pkill -TERM -f "tag gemma_ladder_n128"
pkill -TERM -f "vllm.entrypoints.cli.main serve unsloth/gemma-3-27b-it"
for _ in $(seq 60); do pgrep -f vllm.entrypoints > /dev/null || break; sleep 5; done
pkill -KILL -f "vllm.entrypoints" 2>/dev/null
echo "LANE_STOPPED_FOR_MIXES $(date -u +%FT%TZ)"
while ! grep -q -E "MIX16_DONE|MIX16_FAILED" /workspace/logs/mix16.log; do sleep 60; done
while pgrep -f "vllm.entrypoints" > /dev/null; do sleep 30; done
echo "GEMMA128_STARTED $(date -u +%FT%TZ)"
/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml --models gemma-3-27b --only swarm_gemma_ladder_n128
if [ -f results/.done/gemma-3-27b__swarm_gemma_ladder_n128 ]; then echo "GEMMA128_DONE $(date -u +%FT%TZ)"; else echo "GEMMA128_FAILED $(date -u +%FT%TZ)"; fi
