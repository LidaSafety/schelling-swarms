#!/usr/bin/env bash
# Pairs without the rule (--rules silent) for Qwen3-14B and Qwen3-32B: the base runs the title law needs to predict the
# ruleoff tags (14B with 16 copies; 32B with 4, 8, 16 and 32). Otherwise the fixed prompt of scripts/run_split_grid.sh:
# unaware, no log, no listing, 4 questions, 2 answers each, fixed blocks, seeds 10000-10099, 100 runs per model.
# Usage: scripts/run_ruleoff_pairs.sh [MODEL...]   (default: qwen3-14b qwen3-32b-bf16; each needs its pod:
# python3 scripts/pod.py --model MODEL up). Several invocations may run at once: each writes its own raw file.
set -uo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || set -- qwen3-14b qwen3-32b-bf16
rc=0
for m in "$@"; do
  echo "=== ruleoff_n2 $m start $(date -u +%H:%M:%S)"
  scripts/run_on_pod.sh swarm --models "$m" --channels wiki --log none --awareness unaware --rules silent \
      --listing none --n 100 --tag ruleoff_n2 --workers 32 --agents 2 --questions 4 --f-inv 2 --randomized 0 \
      > "results/ruleoff_n2_$m.log" 2>&1 || rc=$?
  echo "=== ruleoff_n2 $m exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json,sys;print(json.load(open('results/swarm_ruleoff_n2/summary.json'))['models'][sys.argv[1]]['_errors'])" "$m" 2>/dev/null)"
done
exit "$rc"
