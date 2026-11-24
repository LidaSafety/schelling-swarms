#!/usr/bin/env bash
# Page content (registered in the scaling folder's results/predictions_page_content.md before these runs). Pairs on
# the fixed prompt (unaware, rule in force, no page list, no log; 4 questions, 2 answers each, fixed blocks), seeds
# 10000-10099, each played several times (sample offsets), so every arm plays the same quizzes. Before round 1,
# nothing, or one page built from the model's own 64-copy game with the same quiz, sits on the pair's usual round-1
# title for question 1: answers (to question 1, 2 or 3, about a state nobody is asked, or as bare numbers) or
# requests (for question 1 or 2). The pages are the files scripts/make_content_pages.py writes to
# results/page_content/; patches/plant-pages-runner.patch plants them. Needs the model's pod:
# scripts/pod.py --model M up, then wait.
#   scripts/run_page_content_pairs.sh gemma-3-27b     tag plant_content_gemma: 8 arms x 300 games (offsets 0-2)
#   scripts/run_page_content_pairs.sh qwen3-8b        tag plant_content_q8: 6 arms x 400 games (offsets 0-3)
#   TAG=plant_content_pilot N=2 OFFSETS=0 scripts/run_page_content_pairs.sh gemma-3-27b      a pilot: 2 games per arm
# PAGES=<folder> reads the page files from another folder (default results/page_content).
set -uo pipefail
cd "$(dirname "$0")/.."
model=${1:?model: gemma-3-27b or qwen3-8b}
case $model in
  gemma-3-27b) short=gemma; off="0 1 2"
               arms="answers_q1 answers_q3 answers_unasked answers_q1_other requests_q1 requests_q2 answers_q2" ;;
  qwen3-8b)    short=q8; off="0 1 2 3"; arms="answers_q1 answers_unasked requests_q1 requests_q2 answers_q2" ;;
  *) echo "no pages for $model"; exit 1 ;;
esac
tag=${TAG:-plant_content_$short}; n=${N:-100}; pages=${PAGES:-results/page_content}
read -r -a offsets <<< "${OFFSETS:-$off}"
files=()
for arm in $arms; do
  f="$pages/${short}_$arm.json"
  [ -f "$f" ] || { echo "missing $f (scripts/make_content_pages.py writes it)"; exit 1; }
  files+=("$f")
done
echo "=== $tag start $(date -u +%H:%M:%S): $((${#files[@]} + 1)) arms x $n seeds x ${#offsets[@]} offsets"
scripts/run_on_pod.sh swarm --models "$model" --agents 2 --channels wiki --log none --awareness unaware \
    --rules forbidden --listing none --questions 4 --f-inv 2 --randomized 0 --n "$n" --tag "$tag" \
    --plant-file none "${files[@]}" --sample-offset "${offsets[@]}" --workers "${WORKERS:-32}" > "results/$tag.log" 2>&1
rc=$?
echo "=== $tag exit $rc end $(date -u +%H:%M:%S) errors: $(python3 -c "import json;print(json.load(open('results/swarm_$tag/summary.json'))['models']['$model']['_errors'])" 2>/dev/null)"
exit "$rc"
