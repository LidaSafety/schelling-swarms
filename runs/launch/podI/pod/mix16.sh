#!/bin/bash
# Mixed swarms at N = 16 (copied from podH/pod/mix16.sh for pod I, where SKIP_GEMMA=1 and SCB_DIR is the lane's
# checkout, /workspace/schelling-collusion-benchmark, since the lane runs from branch mixed-n16 too), after the lane (or on any H200 set up with setup.sh). Run definitions: entry 20 of
# configs/experiments.yaml on benchmark branch mixed-n16. That branch goes to its own folder on the pod, SCB_DIR
# (default /workspace/scb-mix16), so the lane's checkout is not touched while the lane runs; its results land in
# SCB_DIR/results/swarm_mix16/. Forecasts: research repo analysis/four-round/predictions_mix16.md, committed before
# this starts. From the research repo, with <pod> the pod's ssh address:
#   git -C schelling-collusion-benchmark archive mixed-n16 | ssh <pod> 'mkdir -p /workspace/scb-mix16 && tar -x -C /workspace/scb-mix16'
#   scp schelling-collusion-benchmark/results/logs/podH/pod/mix16.sh <pod>:/workspace/
#
# Qwen3-8B and Gemma 3 27B are served together on the one GPU, as in the hackathon's mixed runs
# (Swarm_schelling_hackathon, scaling/scripts/pod.py --model both): Qwen3-8B on port 8001 with a quarter of the memory,
# Gemma on port 8000. Same vLLM flags as sweep.py and configs/models_h200.yaml otherwise. Then, tag mix16, seed 1
# (games 10000-10099), 100 games each, unaware, rule in force, no listing, no log:
#   Gemma 3 27B alone at N = 16 (Qwen3-8B alone at N = 16 is the main runs, and pod H's plant "none" arm)
#   Gemma shares in eighths: 14 + 2, 12 + 4, 10 + 6, 8 + 8, 6 + 10, 4 + 12 and 2 + 14 copies of Qwen3-8B + Gemma
# 8 x 19.2k = 153.6k calls, about 1.5 hours with the download (Gemma is the slow model). Log lines to watch: MIX16_STARTED, READY ..., MIX16_GEMMA_DONE, MIX16_DONE, MIX16_FAILED.
#   nohup bash /workspace/mix16.sh > /workspace/logs/mix16.log 2>&1 &
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
PY=/workspace/venv/bin/python
L=/workspace/logs
LANE_LOG=${LANE_LOG:-/workspace/logs/lane.log}
SCB_DIR=${SCB_DIR:-/workspace/scb-mix16}
SKIP_GEMMA=${SKIP_GEMMA:-0}   # 1 on pod I: Gemma alone at N = 16 is the lane's swarm_gemma_ladder_n16 (same games)
# Ports clear of RunPod's nginx, which listens on 8001 (and forwards it to 8000), 3001, 7270, 7861, 8081 and 9091.
# On pod H and on pod I's first try, Qwen3-8B could not bind 8001 and every Qwen call reached Gemma: 100% errors.
QPORT=${QPORT:-8101}
GPORT=${GPORT:-8100}
mkdir -p $L
cd "$SCB_DIR" || { echo "MIX16_FAILED no $SCB_DIR: copy benchmark branch mixed-n16 there first"; exit 1; }
fail() { echo "MIX16_FAILED $* $(date -u +%FT%TZ)"; kill ${QP:-} ${GP:-} 2>/dev/null; exit 1; }
echo "MIX16_STARTED $(date -u +%FT%TZ)"

# 1. The runner must have --mix (benchmark branch mixed-n16).
$PY -m scb swarm --help | grep -q -- "--mix" || fail "$SCB_DIR has no swarm --mix; copy benchmark branch mixed-n16 there"

# 2. Weights, downloaded while the lane still runs (network and disk only). Gemma 3 27B is about 55 GB; stop rather than fill the disk.
free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ ! -d "$HF_HOME/hub/models--unsloth--gemma-3-27b-it" ] && [ "$free" -lt 60 ]; then
  fail "only ${free}G free under /workspace; Gemma needs about 55G (weights no longer needed are under $HF_HOME/hub)"
fi
for repo in Qwen/Qwen3-8B unsloth/gemma-3-27b-it; do
  echo "DOWNLOADING $repo $(date -u +%FT%TZ)"
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo'))" 2>&1 | tail -1
done
df -h /workspace | tail -1

# 3. Let pod H's lane finish: its log says so, and no vLLM server is left on the GPU.
if [ -f "$LANE_LOG" ]; then
  while ! grep -q -E "LANE_DONE|LANE_FAILED" "$LANE_LOG"; do sleep 60; done
fi
while pgrep -f "vllm.entrypoints" > /dev/null; do sleep 30; done
echo "GPU_FREE $(date -u +%FT%TZ)"

# 4. Both servers on the one GPU: Qwen3-8B first with a quarter of the memory, then Gemma.
ready() {   # ready <port> <pid> <name>: the server on <port> must list <name> (RunPod's nginx answers on some ports)
  for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q "\"id\": *\"$3\"" && { echo "READY $3 port $1 $(date -u +%FT%TZ)"; return 0; }
    kill -0 "$2" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}
$PY -m vllm.entrypoints.cli.main serve Qwen/Qwen3-8B --served-model-name qwen3-8b --port $QPORT \
  --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.25 > $L/vllm_mix16_qwen3-8b.log 2>&1 &
QP=$!
ready $QPORT $QP qwen3-8b || fail "Qwen3-8B did not start (see $L/vllm_mix16_qwen3-8b.log)"
$PY -m vllm.entrypoints.cli.main serve unsloth/gemma-3-27b-it --served-model-name gemma-3-27b --port $GPORT \
  --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.62 > $L/vllm_mix16_gemma-3-27b.log 2>&1 &
GP=$!
ready $GPORT $GP gemma-3-27b || fail "Gemma 3 27B did not start (see $L/vllm_mix16_gemma-3-27b.log)"

# 5. Each model on its own server; Qwen3 with thinking off, as in models_h200.yaml. SB_EXTRA_BODY is emptied so the
#    Qwen setting is not sent to Gemma.
export SB_API_KEY=EMPTY SB_BASE_URL=http://127.0.0.1:$GPORT/v1 SB_EXTRA_BODY='{}'
export SB_ENDPOINTS='{"qwen3-8b": {"base_url": "http://127.0.0.1:'$QPORT'/v1", "api_key": "EMPTY", "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}, "gemma-3-27b": {"base_url": "http://127.0.0.1:'$GPORT'/v1", "api_key": "EMPTY"}}'
COMMON="--tag mix16 --seed 1 --n 100 --channels wiki --rules forbidden --awareness unaware --listing none --log none --workers 16"
if [ "$SKIP_GEMMA" != 1 ]; then
  $PY -m scb swarm --models gemma-3-27b --agents 16 $COMMON > $L/mix16_gemma.log 2>&1 || fail "Gemma alone (see $L/mix16_gemma.log)"
fi
echo "MIX16_GEMMA_DONE $(date -u +%FT%TZ) SKIP_GEMMA=$SKIP_GEMMA"
MIXES="qwen3-8b:14,gemma-3-27b:2 qwen3-8b:12,gemma-3-27b:4 qwen3-8b:10,gemma-3-27b:6 qwen3-8b:8,gemma-3-27b:8
       qwen3-8b:6,gemma-3-27b:10 qwen3-8b:4,gemma-3-27b:12 qwen3-8b:2,gemma-3-27b:14"
# shellcheck disable=SC2086
$PY -m scb swarm --mix $MIXES $COMMON \
  > $L/mix16_mixes.log 2>&1 || fail "mixes (see $L/mix16_mixes.log)"
kill $QP $GP 2>/dev/null
echo "MIX16_DONE $(date -u +%FT%TZ)"
