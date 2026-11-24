#!/usr/bin/env bash
# Planted notes, a test of precedent (registered in the scaling folder's results/predictions_plant.md before these
# runs). Pairs on the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2 answers each, fixed
# blocks), seeds from 10000, so every arm plays the same quizzes. Before round 1, 0, 4, 16 or 32 notes carrying
# question 1's true value, or 16 notes without a number, sit on the model's favourite round-1 title for question 1
# (scb/swarm.py PLANT_STYLES; patches/plant-notes-runner.patch). Needs the model's pod: scripts/pod.py [--model M] up,
# then wait.
#   scripts/run_plant_pairs.sh gemma-3-27b 200     tags plant_gemma (0, 4, 16, 32) and plant_gemma_none (16)
#   scripts/run_plant_pairs.sh qwen3-8b 400        tags plant_q8 and plant_q8_none
set -uo pipefail
cd "$(dirname "$0")/.."
model=${1:?model}; n=${2:?games per arm}
case $model in gemma-3-27b) short=gemma ;; qwen3-8b) short=q8 ;; *) echo "no plant style for $model"; exit 1 ;; esac
run() {  # tag, extra args
  local tag=$1 rc; shift
  echo "=== $tag start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models "$model" --agents 2 --channels wiki --log none --awareness unaware \
      --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
      --workers "${WORKERS:-32}" "$@" > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc"; exit "$rc"; }
}
run "plant_$short" --plant-notes 0 4 16 32 --plant-kind true
run "plant_${short}_none" --plant-notes 16 --plant-kind none
