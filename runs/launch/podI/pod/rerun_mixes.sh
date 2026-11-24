#!/bin/bash
# Pod I, second try at the mixes (18 Nov, ~21:00 UTC). The first try failed in every game: RunPod's nginx holds port
# 8001, so Qwen3-8B never served and its calls reached Gemma. mix16.sh now uses ports 8101/8100 and checks each server's
# model name. The failed records were moved to results/swarm_mix16_failed_port8001. Then Gemma alone at N = 128, as the
# user asked (mixes before N = 128). Log: /workspace/logs/rerun_mixes.log
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cd /workspace/schelling-collusion-benchmark
echo "RERUN_STARTED $(date -u +%FT%TZ)"
SKIP_GEMMA=1 SCB_DIR=/workspace/schelling-collusion-benchmark bash /workspace/mix16.sh > /workspace/logs/mix16.log 2>&1
tail -1 /workspace/logs/mix16.log
pkill -f "vllm.entrypoints" 2>/dev/null
for _ in $(seq 60); do pgrep -f vllm.entrypoints > /dev/null || break; sleep 5; done
pkill -KILL -f "vllm.entrypoints" 2>/dev/null
echo "GEMMA128_STARTED $(date -u +%FT%TZ)"
/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml --models gemma-3-27b --only swarm_gemma_ladder_n128
if [ -f results/.done/gemma-3-27b__swarm_gemma_ladder_n128 ]; then echo "GEMMA128_DONE $(date -u +%FT%TZ)"; else echo "GEMMA128_FAILED $(date -u +%FT%TZ)"; fi
