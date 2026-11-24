#!/usr/bin/env bash
# The twin text against a crowded page (registered in the scaling folder's results/predictions_twin_page.md before
# these runs). Swarms of 4 Gemma 3 27B copies with the aware or the twin sentence (rule in force, no page list, no log;
# 4 questions, 2 answers each, fixed blocks), seeds 10000-10099, each played twice (sample offsets 1 and 2; offset 0
# is the twin grid's own 4-copy runs). The sentence gives the true size (3 others), or says the swarm has 64 copies
# (patches/stated-size-runner.patch); told 64, nothing or one of item 25's pages (results/transplant/: the 64-copy
# game's round-1 page, its holders' notes only, its seekers' notes only) sits on the usual title for question 1 before
# round 1 (patches/plant-pages-runner.patch). Needs Gemma's pod: scripts/pod.py --model gemma-3-27b up, then wait.
#   scripts/run_twin_page_grid.sh             tags plant_twin_n4 (2 cells) and plant_twin_told64 (8 cells), 200 games each
#   TAG=plant_twin_pilot N=2 OFFSETS=1 scripts/run_twin_page_grid.sh       a pilot: 2 games per cell
# PAGES=<folder> reads the page files from another folder (default results/transplant).
set -uo pipefail
cd "$(dirname "$0")/.."
model=gemma-3-27b; pre=${TAG:-plant_twin}; n=${N:-100}; pages=${PAGES:-results/transplant}
read -r -a offsets <<< "${OFFSETS:-1 2}"
files=()
for arm in page64 holders64 seekers64; do
  [ -f "$pages/$arm.json" ] || { echo "missing $pages/$arm.json (scripts/make_transplant_pages.py writes it)"; exit 1; }
  files+=("$pages/$arm.json")
done
run() {  # tag, extra args
  local tag=$1 rc; shift
  echo "=== $tag start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models "$model" --agents 4 --channels wiki --log none --awareness aware --twin 0 1 \
      --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
      --sample-offset "${offsets[@]}" --workers "${WORKERS:-32}" "$@" > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc"; exit "$rc"; }
}
run "${pre}_n4" --stated-n 0 --plant-file none
run "${pre}_told64" --stated-n 64 --plant-file none "${files[@]}"
