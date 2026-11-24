#!/usr/bin/env bash
# Branches off the fixed prompt (unaware, no log, rule in force, no listing; seeds 10000-10099; 100 runs per cell),
# one design knob at a time, after scripts/run_main_grid.sh and scripts/run_extra_grid.sh:
#   q8_n*      Q=8 questions (f_inv=2, fixed blocks) at N = 2, 4, 8, 16, 32
#   comp_n32   f_inv=4 at N=32, the largest size for the f_inv=4 cells of run_extra_grid.sh (comp_n4..16)
#   r1_n*      R=1: each copy holds its own random Q/f_inv questions (Q=4, f_inv=2) at N = 2, 4, 8, 16, 32
# One tag per cell, so each is saved on its own. Fewer runs in flight at large N: each run sends N calls at once.
set -uo pipefail
cd "$(dirname "$0")/.."
workers() { case $1 in 2|4) echo 32;; 8) echo 24;; 16) echo 12;; *) echo 8;; esac; }
run() {  # tag, workers, extra args; stops the grid when a cell fails (scb exits non-zero, e.g. >2% of its runs failed)
  local tag=$1 w=$2 rc; shift 2
  echo "=== $tag (workers $w) start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models qwen3-8b --channels wiki --log none --awareness unaware --rules forbidden \
      --listing none --n 100 --tag "$tag" --workers "$w" "$@" > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['qwen3-8b']['_errors'])" 2>/dev/null)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc (see results/$tag.log)"; exit "$rc"; }
}
for n in 2 4 8 16 32; do
  run "q8_n$n" "$(workers "$n")" --agents "$n" --questions 8 --f-inv 2 --randomized 0
done
run comp_n32 "$(workers 32)" --agents 32 --questions 4 --f-inv 4 --randomized 0
for n in 2 4 8 16 32; do
  run "r1_n$n" "$(workers "$n")" --agents "$n" --questions 4 --f-inv 2 --randomized 1
done
