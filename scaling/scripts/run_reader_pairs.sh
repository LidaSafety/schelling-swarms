#!/usr/bin/env bash
# The grid test (registered in the scaling folder's results/predictions_grid.md before these runs), the third reader:
# gpt-oss-120b pairs on the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2 answers each,
# fixed blocks) through OpenRouter, with the settings of its or_pairs runs: pinned to one host (Crusoe) without
# fallbacks, reasoning effort low, 6x the 250-token budget. Seeds 10000-10099, sample offsets 0-3 (hosts do not all
# honour a sampling seed, so each offset replays the quizzes, not each reply). Before round 1, nothing, or Gemma's answer
# page from item 26 rewritten by Gemma 3 27B, Qwen3-8B or gpt-oss-120b, moved to gpt-oss-120b's usual round-1 title, the
# state's name (results/reader/, scripts/make_reader_pages.py).
#   scripts/run_reader_pairs.sh               tag plant_grid_gptoss: 4 arms x 400 games
#   TAG=plant_grid_pilot_gptoss N=2 OFFSETS=0 scripts/run_reader_pairs.sh       a pilot: 2 games per arm
# (A first version of this script ran Llama 3.3 70B; its late holders almost never read a page in round 1, so Llama
# serves only as a writer, in item 29.)
set -uo pipefail
cd "$(dirname "$0")/.."
model=openai/gpt-oss-120b
eval "$(python3 scripts/pod.py exports openrouter)"
export SB_EXTRA_BODY='{"provider": {"require_parameters": true, "order": ["Crusoe"], "allow_fallbacks": false}, "usage": {"include": true}, "reasoning": {"effort": "low"}}'
export SB_TOKEN_MULT=6
pages=${PAGES:-results/reader}; tag=${TAG:-plant_grid_gptoss}
files=("$pages/gptoss_by_gemma.json" "$pages/gptoss_by_q8.json" "$pages/gptoss_by_gptoss.json")
for f in "${files[@]}"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
read -r -a offsets <<< "${OFFSETS:-0 1 2 3}"
echo "=== $tag start $(date -u +%H:%M:%S)"
.venv/bin/python -m scb swarm --models "$model" --channels wiki --log none --awareness unaware --rules forbidden \
    --listing none --n "${N:-100}" --tag "$tag" --workers "${WORKERS:-16}" --agents 2 --questions 4 --f-inv 2 \
    --randomized 0 --plant-file none "${files[@]}" --sample-offset "${offsets[@]}" > "results/$tag.log" 2>&1
rc=$?
echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json,sys;print(json.load(open('results/swarm_$tag/summary.json'))['models'][sys.argv[1]]['_errors'])" "$model" 2>/dev/null)"
exit "$rc"
