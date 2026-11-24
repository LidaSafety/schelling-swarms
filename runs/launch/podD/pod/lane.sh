#!/bin/bash
# One H200, one model at a time. Gates hold a block until its predictions are committed locally.
set -u
export HF_HOME=/workspace/hf
cd /workspace/schelling-collusion-benchmark
SW="/workspace/venv/bin/python scripts/sweep.py --models-file configs/models_h200.yaml"
wait_gate() { while [ ! -f "$1" ]; do sleep 20; done; echo "GATE_OPEN $1 $(date -u +%FT%TZ)"; }
echo "LANE_STARTED $(date -u +%FT%TZ)"
$SW --models gemma-3-27b --only wikigame_gemma_small wikigame_gemma_small_crowding
echo "GEMMA_SMALL_DONE $(date -u +%FT%TZ)"
wait_gate /workspace/GATE_14B
$SW --models qwen3-14b --only wikigame_n128 wikigame_purity_s1 wikigame_purity_s24 wikigame_alone wikigame_ruleoff swarm_ruleoff
echo "QWEN14_DONE $(date -u +%FT%TZ)"
$SW --models qwen3-4b --only wikigame_alone
$SW --models qwen3-8b --only wikigame_ruleoff
echo "QWEN_4_8_DONE $(date -u +%FT%TZ)"
rm -rf /workspace/hf/hub/models--Qwen--Qwen3-14B /workspace/hf/hub/models--Qwen--Qwen3-4B /workspace/hf/hub/models--Qwen--Qwen3-8B
$SW --models qwen3-32b-bf16 --only swarm_ruleoff
echo "QWEN32_DONE $(date -u +%FT%TZ)"
wait_gate /workspace/GATE_GEMMA2
$SW --models gemma-3-27b --only wikigame_gemma_large
echo "LANE_DONE $(date -u +%FT%TZ)"
