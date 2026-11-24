#!/usr/bin/env bash
# Extra cells, run after the main grid (scripts/run_main_grid.sh) so they never compete for the GPU:
#   composition: f_inv=4, each copy holds 1 of the 4 answers; N = 4, 8, 16 -> N/4 holders per question (main grid: N/2)
#   defense: the benchmark's warning note (scb/defense.py "warning") at N = 2, 4, 8, 16 with f_inv=2
# Unaware and aware, no log, rule in force, no listing, seeds 10000-10099: every run has a paired main-grid run.
# (The runs in results/ also have the twin arm, --twin 1, which commit 107ae64 removed.)
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done; fi
run() {  # tag, workers, extra args
  local tag=$1 w=$2; shift 2
  echo "=== $tag (workers $w) start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models qwen3-8b --channels wiki --questions 4 --randomized 0 --log none \
      --awareness aware unaware --rules forbidden --listing none --n 100 --tag "$tag" --workers "$w" "$@" \
      > "results/$tag.log" 2>&1
  echo "=== $tag exit $? end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['qwen3-8b']['_errors'])" 2>/dev/null)"
}
run comp_n4 32 --agents 4 --f-inv 4
run comp_n8 24 --agents 8 --f-inv 4
run comp_n16 12 --agents 16 --f-inv 4
for n in 2 4 8 16; do
  w=$(( n <= 4 ? 32 : (n == 8 ? 24 : 12) ))
  run "def_n$n" "$w" --agents "$n" --f-inv 2 --defenses warning
done
