#!/usr/bin/env bash
# The grid test (registered in the scaling folder's results/predictions_grid.md before these runs), the pod readers:
# Gemma 3 27B or Qwen3-8B pairs on the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2
# answers each, fixed blocks), seeds 10000-10099 with sample offsets new to these quizzes. Before round 1, nothing, or
# the model's answer page from item 26 rewritten note by note by Gemma 3 27B, Qwen3-8B or gpt-oss-120b
# (results/paraphrase/), sits on the pair's usual round-1 title for question 1. Needs the model's pod.
#   scripts/run_grid_pairs.sh gemma-3-27b     tag plant_grid_gemma: 4 arms x 300 games (offsets 6-8)
#   scripts/run_grid_pairs.sh qwen3-8b        tag plant_grid_q8: 4 arms x 400 games (offsets 8-11)
#   TAG=plant_grid_pilot N=2 OFFSETS=6 scripts/run_grid_pairs.sh gemma-3-27b        a pilot: 2 games per arm
set -uo pipefail
cd "$(dirname "$0")/.."
model=${1:?model: gemma-3-27b or qwen3-8b}
case $model in
  gemma-3-27b) short=gemma; off="6 7 8" ;;
  qwen3-8b)    short=q8; off="8 9 10 11" ;;
  *) echo "no pages for $model"; exit 1 ;;
esac
tag=${TAG:-plant_grid_$short}; n=${N:-100}; para=${PARA:-results/paraphrase}
read -r -a offsets <<< "${OFFSETS:-$off}"
files=("$para/${short}_by_gemma.json" "$para/${short}_by_q8.json" "$para/${short}_by_gptoss.json")
for f in "${files[@]}"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
echo "=== $tag start $(date -u +%H:%M:%S): 4 arms x $n seeds x ${#offsets[@]} offsets"
scripts/run_on_pod.sh swarm --models "$model" --agents 2 --channels wiki --log none --awareness unaware \
    --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
    --plant-file none "${files[@]}" --sample-offset "${offsets[@]}" --workers "${WORKERS:-32}" > "results/$tag.log" 2>&1
rc=$?
echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
exit "$rc"
