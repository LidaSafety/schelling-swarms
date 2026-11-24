#!/usr/bin/env bash
# Pairs, with and without the rule, for the six OpenRouter models of the or_<model> tags (16 copies): the base runs
# the title law needs to predict those cells. The same settings as those tags, read from their rows: OpenRouter, each
# model pinned to the same host without fallbacks, reasoning off where the model allows it (gpt-oss-120B and GLM-5.3
# Flash at low effort, with 6x the 250-token budget). Fixed prompt otherwise: unaware, no log, no listing, 4 questions,
# 2 answers each, seeds 10000-10099, 100 runs per arm. OpenRouter hosts do not all honour a sampling seed, so the runs
# repeat the quizzes, not each reply.
# Usage: scripts/run_api_pairs.sh pilot|full [MODEL...]   (pilot: 2 runs per arm, tag or_pairs_pilot)
# WORKERS (default 16) and TAG override the concurrency and the tag: GLM-5.3 Flash's host rate-limits 16 at once,
# so it ran with WORKERS=2 TAG=or_pairs_glm. AGENTS (default 2) and RULES (default "forbidden silent") override the
# swarm sizes and the rules: DeepSeek V4.1 Flash without the rule with 32 and 64 copies (registered in the scaling
# folder's results/predictions_deepseek_large.md) ran with AGENTS="32 64" RULES=silent TAG=deepseek_large WORKERS=2.
set -uo pipefail
cd "$(dirname "$0")/.."
mode=${1:?pilot or full}; shift
eval "$(python3 scripts/pod.py exports openrouter)"
settings() {  # model -> "host token_mult reasoning-json"
  case $1 in
    openai/gpt-oss-120b)          echo 'Crusoe 6 {"effort": "low"}';;
    z-ai/glm-5.3-flash)           echo 'Parasail 6 {"effort": "low"}';;
    deepseek/deepseek-v4.1-flash) echo 'DeepInfra 1 {"effort": "none"}';;
    qwen/qwen3-235b-a22b-2507)    echo 'DeepInfra 1 -';;
    qwen/qwen3.5-27b|qwen/qwen3-32b) echo 'DeepInfra 1 {"enabled": false}';;
    *) return 1;;
  esac
}
[ $# -gt 0 ] || set -- openai/gpt-oss-120b z-ai/glm-5.3-flash deepseek/deepseek-v4.1-flash qwen/qwen3-235b-a22b-2507 \
                       qwen/qwen3.5-27b qwen/qwen3-32b
if [ "$mode" = pilot ]; then n=2; tag=${TAG:-or_pairs}_pilot; else n=100; tag=${TAG:-or_pairs}; fi
agents=${AGENTS:-2}; rules=${RULES:-forbidden silent}
rc=0
for m in "$@"; do
  mrc=0
  read -r host mult reason <<< "$(settings "$m")" || { echo "unknown model $m"; exit 1; }
  export SB_EXTRA_BODY="$(python3 -c 'import json,sys
h, r = sys.argv[1], sys.argv[2]
d = {"provider": {"require_parameters": True, "order": [h], "allow_fallbacks": False}, "usage": {"include": True}}
if r != "-": d["reasoning"] = json.loads(r)
print(json.dumps(d))' "$host" "$reason")"
  export SB_TOKEN_MULT="$mult"
  log="results/${tag}_${m//\//_}.log"
  echo "=== $tag $m ($host, x$mult, reasoning $reason) start $(date -u +%H:%M:%S)"
  .venv/bin/python -m scb swarm --models "$m" --channels wiki --log none --awareness unaware --rules $rules \
      --listing none --n "$n" --tag "$tag" --workers "${WORKERS:-16}" --agents $agents --questions 4 --f-inv 2 --randomized 0 \
      > "$log" 2>&1 || mrc=$?
  [ "$mrc" -eq 0 ] || rc=$mrc
  echo "=== $tag $m exit $mrc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json,sys;print(json.load(open('results/swarm_$tag/summary.json'))['models'][sys.argv[1]]['_errors'])" "$m" 2>/dev/null)"
done
exit "$rc"
