#!/bin/bash
# Pod I, 19 November: the capability quiz (entry 25) for the vLLM models, after Gemma 3's no-rule run. Runs from
# /workspace/scb-cap (benchmark branch claude-api), so the running checkout is not touched. Log: logs/capability.log
set -u
export HF_HOME=/workspace/hf PATH=/workspace/venv/bin:$PATH CUDA_HOME=/usr/local/cuda
PY=/workspace/venv/bin/python
echo "CAP_STARTED $(date -u +%FT%TZ)"
for repo in Qwen/Qwen3-4B Qwen/Qwen3-14B Qwen/Qwen3-32B; do
  echo "DOWNLOADING $repo $(date -u +%FT%TZ)"
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo'))" 2>&1 | tail -1
done
df -h /workspace | tail -1
while ! grep -qE "GEMMA_RULEOFF_(DONE|FAILED)" /workspace/logs/gemma_ruleoff.log; do sleep 30; done
echo "GPU_FREE_WAIT_DONE $(date -u +%FT%TZ)"
cd /workspace/scb-cap
$PY scripts/sweep.py --models-file configs/models_h200.yaml --models qwen3-4b qwen3-8b qwen3-14b qwen3-32b-bf16 gemma-3-27b --only capability_mmlupro
n=$(ls results/.done 2>/dev/null | grep -c capability_mmlupro)
echo "CAP_DONE $n of 5 $(date -u +%FT%TZ)"
