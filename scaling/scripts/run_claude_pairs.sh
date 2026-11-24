#!/usr/bin/env bash
# Pairs, with and without the rule, for the four Claude models of the claude_* tags on the aaai branch (16 copies):
# Haiku 5.5, Haiku 4.5, Sonnet 5.5 and Opus 4.6, through Anthropic's Messages API with the official SDK (scb/llm.py,
# SB_BACKEND=anthropic). As in those runs: 250 tokens per call and no thinking or sampling options (their calls used no
# thinking tokens). Fixed prompt otherwise: unaware, no log, no listing, 4 questions, 2 answers each,
# seeds 10000-10099, 100 runs per arm. The API has no sampling seed, so the runs repeat the quizzes, not each reply.
# Usage: scripts/run_claude_pairs.sh pilot|full [MODEL...]   (pilot: 2 runs per arm, tag claude_pairs_pilot)
set -uo pipefail
cd "$(dirname "$0")/.."
mode=${1:?pilot or full}; shift
eval "$(python3 scripts/pod.py exports anthropic)"
export SB_EXTRA_BODY='{}' SB_TOKEN_MULT=1
[ $# -gt 0 ] || set -- claude-haiku-5-5 claude-haiku-4-5 claude-sonnet-5-5 claude-opus-4-6
if [ "$mode" = pilot ]; then n=2; tag=claude_pairs_pilot; else n=100; tag=${TAG:-claude_pairs}; fi
rc=0
for m in "$@"; do
  mrc=0
  echo "=== $tag $m start $(date -u +%H:%M:%S)"
  .venv/bin/python -m scb swarm --models "$m" --channels wiki --log none --awareness unaware --rules forbidden silent \
      --listing none --n "$n" --tag "$tag" --workers "${WORKERS:-8}" --agents 2 --questions 4 --f-inv 2 --randomized 0 \
      > "results/${tag}_$m.log" 2>&1 || mrc=$?
  [ "$mrc" -eq 0 ] || rc=$mrc
  echo "=== $tag $m exit $mrc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json,sys;print(json.load(open('results/swarm_$tag/summary.json'))['models'][sys.argv[1]]['_errors'])" "$m" 2>/dev/null)"
done
exit "$rc"
