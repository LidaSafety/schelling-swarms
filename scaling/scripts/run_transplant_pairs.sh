#!/usr/bin/env bash
# Transplanted pages (registered in the scaling folder's results/predictions_transplant.md before these runs). Pairs of
# Gemma 3 27B on the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2 answers each, fixed
# blocks), seeds 10000-10099, each played three times (sample offsets 0, 1 and 2), so every arm plays the same quizzes.
# Before round 1, nothing, or one recorded round-1 page (the 64-copy game's page, its holders' notes, its seekers'
# notes, or the 32-copy game's page) sits on the pair's usual round-1 title for question 1. The pages are the files
# scripts/make_transplant_pages.py writes to results/transplant/; patches/plant-pages-runner.patch plants them.
# Needs Gemma's pod: scripts/pod.py --model gemma-3-27b up, then wait.
#   scripts/run_transplant_pairs.sh                       tag plant_transplant: 5 arms x 300 games
#   TAG=plant_transplant_pilot N=2 OFFSETS=0 scripts/run_transplant_pairs.sh      a pilot: 2 games per arm
# PAGES=<folder> reads the page files from another folder (default results/transplant).
set -uo pipefail
cd "$(dirname "$0")/.."
model=gemma-3-27b; tag=${TAG:-plant_transplant}; n=${N:-100}; pages=${PAGES:-results/transplant}
read -r -a offsets <<< "${OFFSETS:-0 1 2}"
files=()
for arm in page64 holders64 seekers64 page32; do
  [ -f "$pages/$arm.json" ] || { echo "missing $pages/$arm.json (scripts/make_transplant_pages.py writes it)"; exit 1; }
  files+=("$pages/$arm.json")
done
echo "=== $tag start $(date -u +%H:%M:%S): 5 arms x $n seeds x ${#offsets[@]} offsets"
scripts/run_on_pod.sh swarm --models "$model" --agents 2 --channels wiki --log none --awareness unaware \
    --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
    --plant-file none "${files[@]}" --sample-offset "${offsets[@]}" --workers "${WORKERS:-32}" > "results/$tag.log" 2>&1
rc=$?
echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
exit "$rc"
