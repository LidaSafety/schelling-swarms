#!/bin/bash
# Pod J (22 Nov 2026): swarms mixing Qwen3-4B, Qwen3-8B and Gemma 3 27B, for the paper's Figure 14 (fig:simplex, App. F),
# whose 4B mixes were predictions only. Run definitions: entry 26 of configs/experiments.yaml on benchmark branch
# mixed-3way, unpacked to SCB_DIR (default /workspace/schelling-collusion-benchmark); results land in
# SCB_DIR/results/swarm_mix3/. Forecasts registered before this starts: research repo
# analysis/four-round/predictions_mix3.md (all mixes) and, for four of them, the hackathon repo's
# scaling/results/predictions_mix3.md (ykazemi, branch scaling, acba44a; their game counts are kept: 200 for 6:8:2
# and 8:8:0). From the research repo, with <pod> the pod's ssh address:
#   git -C schelling-collusion-benchmark archive mixed-3way | ssh <pod> 'tar -x -C /workspace/schelling-collusion-benchmark'
#   scp schelling-collusion-benchmark/results/logs/podJ/pod/mix3.sh <pod>:/workspace/
#   ssh <pod> 'nohup bash /workspace/mix3.sh > /workspace/logs/mix3.log 2>&1 &'
#
# The three models are served together on one H200, as mix16.sh served two: Qwen3-4B (port 8102, 12% of memory),
# Qwen3-8B (8101, 20%) and Gemma 3 27B (8100, 58%). Same vLLM flags as sweep.py and configs/models_h200.yaml otherwise.
# Every mix: tag mix3, seed 1 (games from 10000), unaware, rule in force, no listing, no log, 2 answers per copy, seats
# in the order 4B, 8B, Gemma, each model an even number of copies, so it holds answers in both halves of the quiz.
# Order (most needed first): ykazemi's four registered mixes, then the rest of the inside of the triangle, then the
# 4B + 8B edge, then the 4B + Gemma edge. Gemma sets the time: about 2.5-3 h in all (pod I: 79 min for the 8B + Gemma edge).
# Log lines to watch: MIX3_STARTED, READY ..., BLOCK_DONE <name>, MIX3_DONE, MIX3_FAILED.
set -u
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH
export CUDA_HOME=/usr/local/cuda
PY=/workspace/venv/bin/python
L=/workspace/logs
SCB_DIR=${SCB_DIR:-/workspace/schelling-collusion-benchmark}
# Ports clear of RunPod's nginx, which listens on 8001 (and forwards it to 8000), 3001, 7270, 7861, 8081 and 9091.
P4=${P4:-8102}
P8=${P8:-8101}
PG=${PG:-8100}
mkdir -p $L
cd "$SCB_DIR" || { echo "MIX3_FAILED no $SCB_DIR"; exit 1; }
fail() { echo "MIX3_FAILED $* $(date -u +%FT%TZ)"; kill ${S4:-} ${S8:-} ${SG:-} 2>/dev/null; exit 1; }
echo "MIX3_STARTED $(date -u +%FT%TZ)"

# 1. The runner must have --mix.
$PY -m scb swarm --help | grep -q -- "--mix" || fail "$SCB_DIR has no swarm --mix"

# 2. Weights (setup.sh downloads them; this only waits for that and checks).
while ! grep -q DOWNLOAD_OK $L/download.log 2>/dev/null; do sleep 30; done
for repo in Qwen/Qwen3-4B Qwen/Qwen3-8B unsloth/gemma-3-27b-it; do
  $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$repo'))" 2>&1 | tail -1
done
while pgrep -f "vllm.entrypoints" > /dev/null; do sleep 30; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "GPU_FREE $(date -u +%FT%TZ)"

# 3. The three servers on the one GPU, smallest first.
ready() {   # ready <port> <pid> <name>: the server on <port> must list <name> (RunPod's nginx answers on some ports)
  for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q "\"id\": *\"$3\"" && { echo "READY $3 port $1 $(date -u +%FT%TZ)"; return 0; }
    kill -0 "$2" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}
serve() {   # serve <hf id> <name> <port> <memory share>
  $PY -m vllm.entrypoints.cli.main serve "$1" --served-model-name "$2" --port "$3" \
    --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization "$4" > $L/vllm_mix3_$2.log 2>&1 &
}
serve Qwen/Qwen3-4B qwen3-4b $P4 0.12; S4=$!
ready $P4 $S4 qwen3-4b || fail "Qwen3-4B did not start (see $L/vllm_mix3_qwen3-4b.log)"
serve Qwen/Qwen3-8B qwen3-8b $P8 0.20; S8=$!
ready $P8 $S8 qwen3-8b || fail "Qwen3-8B did not start (see $L/vllm_mix3_qwen3-8b.log)"
serve unsloth/gemma-3-27b-it gemma-3-27b $PG 0.58; SG=$!
ready $PG $SG gemma-3-27b || fail "Gemma 3 27B did not start (see $L/vllm_mix3_gemma-3-27b.log)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 4. Each model on its own server; Qwen3 with thinking off, as in models_h200.yaml. SB_EXTRA_BODY is emptied so the
#    Qwen setting is not sent to Gemma.
export SB_API_KEY=EMPTY SB_BASE_URL=http://127.0.0.1:$PG/v1 SB_EXTRA_BODY='{}'
QX='"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}'
export SB_ENDPOINTS='{"qwen3-4b": {"base_url": "http://127.0.0.1:'$P4'/v1", "api_key": "EMPTY", '"$QX"'},
 "qwen3-8b": {"base_url": "http://127.0.0.1:'$P8'/v1", "api_key": "EMPTY", '"$QX"'},
 "gemma-3-27b": {"base_url": "http://127.0.0.1:'$PG'/v1", "api_key": "EMPTY"}}'
$PY -c "import json, os; print(sorted(json.loads(os.environ['SB_ENDPOINTS'])))" || fail "SB_ENDPOINTS is not valid JSON"

# 5. No game before the forecasts are committed: wait for /workspace/GO, made after that commit.
echo "WAITING_FOR_GO $(date -u +%FT%TZ)"
while [ ! -f /workspace/GO ]; do sleep 15; done
echo "GO $(date -u +%FT%TZ) $(cat /workspace/GO)"
COMMON="--tag mix3 --seed 1 --channels wiki --rules forbidden --awareness unaware --listing none --log none --workers 16"
block() {   # block <name> <games> <mix> [<mix> ...]
  local name=$1 n=$2; shift 2
  echo "BLOCK_STARTED $name $(date -u +%FT%TZ)"
  # shellcheck disable=SC2086
  $PY -m scb swarm --mix "$@" --n $n $COMMON > $L/mix3_$name.log 2>&1 || fail "block $name (see $L/mix3_$name.log)"
  echo "BLOCK_DONE $name $(date -u +%FT%TZ)"
}
A=qwen3-4b; B=qwen3-8b; C=gemma-3-27b
# ykazemi's four registered mixes (hackathon repo scaling/results/predictions_mix3.md), with their game counts
block registered200 200 "$A:6,$B:8,$C:2" "$A:8,$B:8"
block registered100 100 "$A:4,$B:4,$C:8" "$A:8,$C:8"
# the rest of the inside of the triangle, and the even three-way mix (exact only at N = 12)
block inside 100 "$A:6,$B:6,$C:4" "$A:8,$B:4,$C:4" "$A:4,$B:8,$C:4" "$A:6,$B:4,$C:6" "$A:4,$B:6,$C:6" "$A:4,$B:4,$C:4"
# the 4B + 8B edge, 8B share in eighths (8 + 8 is above)
block edge48 100 "$A:14,$B:2" "$A:12,$B:4" "$A:10,$B:6" "$A:6,$B:10" "$A:4,$B:12" "$A:2,$B:14"
# the 4B + Gemma edge, Gemma share in eighths, as pod I's 8B + Gemma edge (8 + 8 is above)
block edge4g 100 "$A:14,$C:2" "$A:12,$C:4" "$A:10,$C:6" "$A:6,$C:10" "$A:4,$C:12" "$A:2,$C:14"
kill $S4 $S8 $SG 2>/dev/null
echo "MIX3_DONE $(date -u +%FT%TZ)"
