#!/usr/bin/env bash
# The twin text at scale: does telling copies that the others are copies of them change how they react to crowded
# pages (precedent), and does that show at large N? Registered in the scaling folder's results/predictions_twin.md
# before these runs. The fixed prompt but the other-copies sentence: the aware text and the twin text in one tag
# (--awareness aware --twin 0 1; patches/twin-text-runner.patch), seeds 10000-10099, so both texts play the same
# quizzes. No log, no page list, 4 questions, 2 answers each, fixed blocks. Needs the model's pod:
# scripts/pod.py [--model M] up, then wait.
#   MODEL=gemma-3-27b SUFFIX=_gemma scripts/run_twin_grid.sh
#       tags twin_n2_gemma ... twin_n64_gemma (rule in force) and twinoff_n16_gemma (no rule)
#   MODEL=qwen3-8b SUFFIX=_q8 SIZES=64 OFF= scripts/run_twin_grid.sh      tag twin_n64_q8
set -uo pipefail
cd "$(dirname "$0")/.."
model=${MODEL:-gemma-3-27b}; sfx=${SUFFIX:-_gemma}; sizes=${SIZES:-2 4 8 16 32 64}; off=${OFF-16}
workers() { case $1 in 2|4) echo 32;; 8) echo 24;; 16) echo 12;; *) echo 8;; esac; }   # as run_branch_grid.sh
run() {  # tag, copies, rule
  local tag=$1$sfx rc
  echo "=== $tag start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models "$model" --channels wiki --log none --awareness aware --twin 0 1 \
      --rules "$3" --listing none --n 100 --tag "$tag" --workers "$(workers "$2")" --agents "$2" --questions 4 \
      --f-inv 2 --randomized 0 > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc"; exit "$rc"; }
}
for n in $sizes; do run "twin_n$n" "$n" forbidden; done
for n in $off; do run "twinoff_n$n" "$n" silent; done
