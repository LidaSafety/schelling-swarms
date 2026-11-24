#!/usr/bin/env bash
# Q=8 questions with fewer answers per copy, on the fixed prompt of scripts/run_branch_grid.sh (unaware, no log, rule in
# force, no listing; seeds 10000-10099; 100 runs per cell; fixed blocks):
#   q8f4_n4, q8f4_n16   f_inv=4: each copy holds 2 of the 8 answers (a quarter of the quiz); 1 and 4 copies hold each
#   q8f8_n8, q8f8_n32   f_inv=8: each copy holds 1 of the 8 answers (an eighth);            1 and 4 copies hold each
# With q8_n2..32 (f_inv=2, 4 of 8 each, half the quiz) this separates the share of the quiz a copy holds from the number
# of answers it holds, and gives three cells with 4 copies holding each answer: q8_n8, q8f4_n16, q8f8_n32 (4, 12 and 28
# copies without it). The smallest runs of each split (q8_n2, q8f4_n4, q8f8_n8) are the title law's base.
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
run q8f4_n4 "$(workers 4)" --agents 4 --questions 8 --f-inv 4 --randomized 0
run q8f8_n8 "$(workers 8)" --agents 8 --questions 8 --f-inv 8 --randomized 0
run q8f4_n16 "$(workers 16)" --agents 16 --questions 8 --f-inv 4 --randomized 0
run q8f8_n32 "$(workers 32)" --agents 32 --questions 8 --f-inv 8 --randomized 0
