#!/usr/bin/env bash
# Does a copy's title choice follow how many answers it holds, or how many values its notes list? The notes list the
# held answers plus filler states that are never asked (2 by default), so holding 1 answer means seeing 3 values and
# holding 2 means seeing 4. These cells cross the two (Qwen3-8B, fixed prompt of scripts/run_split_grid.sh, 4 questions,
# seeds 10000-10099, 100 runs per cell; needs patches/filler-count-runner.patch for --fillers):
#   fill1_n2  2 answers + 1 filler = 3 values (pairs)        fill0_n2  2 answers, no filler = 2 values (pairs)
#   fill3_n4  1 answer + 3 fillers = 4 values (4 copies)     fill1_n4  1 answer + 1 filler  = 2 values (4 copies)
# With the existing main_n2 (2 + 2 = 4 values) and comp_n4 (1 + 2 = 3 values).
# MODEL and SUFFIX run the same cells for another model: Gemma 3 27B ran with MODEL=gemma-3-27b SUFFIX=_gemma (tags
# fill1_n2_gemma, ...), on its pod. Its gemma_n2 is the 2 + 2 = 4-value pair cell; its 4-copy runs hold 2 answers each,
# so the 1 + 2 = 3-value cell (comp_n4 for Qwen3-8B) runs too, as comp_n4<SUFFIX>.
set -uo pipefail
cd "$(dirname "$0")/.."
model=${MODEL:-qwen3-8b}; sfx=${SUFFIX:-}
run() {  # tag, extra args
  local tag=$1$sfx rc; shift
  echo "=== $tag start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models "$model" --channels wiki --log none --awareness unaware --rules forbidden \
      --listing none --n 100 --tag "$tag" --workers 32 --questions 4 --randomized 0 "$@" > "results/$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
  [ "$rc" -eq 0 ] || { echo "=== stopping: $tag exited with $rc"; exit "$rc"; }
}
run fill1_n2 --agents 2 --f-inv 2 --fillers 1
run fill0_n2 --agents 2 --f-inv 2 --fillers 0
run fill3_n4 --agents 4 --f-inv 4 --fillers 3
run fill1_n4 --agents 4 --f-inv 4 --fillers 1
[ "$model" = qwen3-8b ] || run comp_n4 --agents 4 --f-inv 4 --fillers 2
