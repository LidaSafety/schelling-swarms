#!/usr/bin/env bash
# The own-style test (registered in the scaling folder's results/predictions_paraphrase.md before these runs). Pairs
# on the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2 answers each, fixed blocks), seeds
# 10000-10099, each played several times (sample offsets). Before round 1, nothing, or the answer page of the model's
# own 64-copy game with the same quiz, sits on the pair's usual round-1 title for question 1: as written
# (results/page_content/), rewritten note by note by the same model, or by the other model (results/paraphrase/,
# scripts/make_paraphrase_pages.py). Needs the model's pod: scripts/pod.py --model M up, then wait.
#   scripts/run_paraphrase_pairs.sh gemma-3-27b     tag plant_style_gemma: 4 arms x 300 games (offsets 0-2)
#   scripts/run_paraphrase_pairs.sh qwen3-8b        tag plant_style_q8: 4 arms x 400 games (offsets 0-3)
#   TAG=plant_style_pilot N=2 OFFSETS=0 scripts/run_paraphrase_pairs.sh gemma-3-27b       a pilot: 2 games per arm
# PAGES and PARA read the page files from other folders (defaults results/page_content and results/paraphrase).
set -uo pipefail
cd "$(dirname "$0")/.."
model=${1:?model: gemma-3-27b or qwen3-8b}
case $model in
  gemma-3-27b) short=gemma; other=q8; off="0 1 2" ;;
  qwen3-8b)    short=q8; other=gemma; off="0 1 2 3" ;;
  *) echo "no pages for $model"; exit 1 ;;
esac
tag=${TAG:-plant_style_$short}; n=${N:-100}; pages=${PAGES:-results/page_content}; para=${PARA:-results/paraphrase}
read -r -a offsets <<< "${OFFSETS:-$off}"
files=("$pages/${short}_answers_q1.json" "$para/${short}_by_${short}.json" "$para/${short}_by_${other}.json")
for f in "${files[@]}"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
echo "=== $tag start $(date -u +%H:%M:%S): 4 arms x $n seeds x ${#offsets[@]} offsets"
scripts/run_on_pod.sh swarm --models "$model" --agents 2 --channels wiki --log none --awareness unaware \
    --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
    --plant-file none "${files[@]}" --sample-offset "${offsets[@]}" --workers "${WORKERS:-32}" > "results/$tag.log" 2>&1
rc=$?
echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
exit "$rc"
