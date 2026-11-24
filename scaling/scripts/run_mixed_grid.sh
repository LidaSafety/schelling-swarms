#!/usr/bin/env bash
# Mixed swarms on the fixed prompt (unaware, no log, rule in force, no listing; Q=4, f_inv=2, fixed blocks; seeds
# 10000-10099; 100 runs per cell): copies of Qwen3-8B and Gemma 3 27B on one wiki. Needs both pods running:
#   python3 scripts/pod.py up; python3 scripts/pod.py --model gemma-3-27b up   (then `wait` for each)
# Steps, in this order:
#   gemma_n2          Gemma's 2-copy runs: the base of the prediction (Qwen's base runs are main_n2)
#   prediction        scripts/predict_mixed.py for every mix, written to results/predict_mixed.json before any mixed run
#   mix_n4            2 Qwen + 2 Gemma
#   mix_n8            4 + 4, 6 + 2 and 2 + 6. Seats are filled in the order of the mix, so each model holds both halves
#                     of the answers (fixed blocks: copy k holds block (k+1) mod 2)
#   gemma_n4, _n8     Gemma alone at the same sizes (Qwen alone: main_n4 and main_n8)
# A cell that fails (scb exits non-zero, e.g. more than 2% of its runs failed) stops the grid.
set -uo pipefail
cd "$(dirname "$0")/.."
MIX_4="qwen3-8b:2,gemma-3-27b:2"
MIXES_8="qwen3-8b:4,gemma-3-27b:4 qwen3-8b:6,gemma-3-27b:2 qwen3-8b:2,gemma-3-27b:6"
run() {  # tag, workers, extra args
  local tag=$1 w=$2 rc; shift 2
  echo "=== $tag (workers $w) start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --channels wiki --log none --awareness unaware --rules forbidden --listing none \
      --questions 4 --f-inv 2 --randomized 0 --n 100 --tag "$tag" --workers "$w" "$@" > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc (see results/$tag.log)"; exit "$rc"; }
}
done_tag() { [ -s "results/swarm_$1/summary.json" ] && echo "=== $1 already done, skipped"; }   # a rerun resumes
done_tag gemma_n2 || run gemma_n2 32 --models gemma-3-27b --agents 2
if [ -s results/predict_mixed.json ]; then
  echo "=== prediction already written, kept"        # it must stay the one made before any mixed run
else
  echo "=== prediction start $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2046
  .venv/bin/python scripts/predict_mixed.py results --mix "$MIX_4" $(for m in $MIXES_8; do echo --mix "$m"; done) \
      > results/predict_mixed.log 2>&1 || { echo "=== prediction failed (see results/predict_mixed.log)"; exit 1; }
  echo "=== prediction written $(date -u +%H:%M:%S)"
fi
# the mixes also need the Qwen3-8B pod, which may have been started only after Gemma's
python3 scripts/pod.py wait > results/mixed_wait_qwen.log 2>&1 || { echo "=== Qwen pod not ready (see results/mixed_wait_qwen.log)"; exit 1; }
echo "=== Qwen pod ready $(date -u +%H:%M:%S)"
done_tag mix_n4 || run mix_n4 32 --mix "$MIX_4"
# shellcheck disable=SC2086
done_tag mix_n8 || run mix_n8 24 --mix $MIXES_8
done_tag gemma_n4 || run gemma_n4 32 --models gemma-3-27b --agents 4
done_tag gemma_n8 || run gemma_n8 24 --models gemma-3-27b --agents 8
