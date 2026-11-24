#!/usr/bin/env bash
# The scaling grid on the pod: wiki, rule in force, no listing, no log, Q=4, fixed blocks; unaware and aware, as in
# scale_unaware and scale_aware (configs/hackathon_scaling.yaml); 100 seeds (10000-10099, the stage-0 seeds);
# one tag per N so each size is saved on its own. Fewer runs in flight at large N: each run sends N calls at once.
# The runs in results/ also have a third arm, --twin 1 ("copies of you"), which commit 107ae64 removed.
set -uo pipefail
cd "$(dirname "$0")/.."
for spec in "2 32" "4 32" "8 24" "16 12" "32 8"; do
  set -- $spec
  echo "=== N=$1 (workers $2) start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models qwen3-8b --channels wiki --agents "$1" --questions 4 --f-inv 2 --randomized 0 \
      --log none --awareness aware unaware --rules forbidden --listing none --n 100 --tag "main_n$1" --workers "$2" \
      > "results/main_n$1.log" 2>&1
  echo "=== N=$1 exit $? end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_main_n$1/summary.json'))['models']['qwen3-8b']['_errors'])" 2>/dev/null)"
done
