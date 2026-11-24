#!/bin/bash
# Pod setup: environment as in the earlier pod sessions (vllm 0.32.0), then model downloads in the order they are needed.
set -u
export HF_HOME=/workspace/hf
mkdir -p /workspace/logs /workspace/hf
cd /workspace
{
  echo "SETUP_STARTED $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  uv venv /workspace/venv --python 3.12 || python3 -m venv /workspace/venv
  uv pip install --python /workspace/venv/bin/python -c /workspace/pip_freeze_podC.txt vllm==0.32.0 openai pyyaml pytest \
    || /workspace/venv/bin/python -m pip install -c /workspace/pip_freeze_podC.txt vllm==0.32.0 openai pyyaml pytest
  /workspace/venv/bin/python -c "import vllm, torch, transformers, openai; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'openai', openai.__version__)"
  cd /workspace/schelling-collusion-benchmark && /workspace/venv/bin/python -m pytest -q tests 2>&1 | tail -3
  echo "ENV_OK $(date -u +%FT%TZ)"
} > /workspace/logs/setup.log 2>&1
{
  for repo in unsloth/gemma-3-27b-it Qwen/Qwen3-14B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-32B; do
    echo "DOWNLOADING $repo $(date -u +%FT%TZ)"
    /workspace/venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo'))" 2>&1 | tail -1
    df -h /workspace | tail -1
  done
  echo "DOWNLOAD_OK $(date -u +%FT%TZ)"
} > /workspace/logs/download.log 2>&1
