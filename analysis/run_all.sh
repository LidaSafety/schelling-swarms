#!/bin/bash
# Rebuild every write-up the paper's numbers come from, in dependency order, then compare each with the copy in
# expected/ (the files the paper's source comment block cites). Run from anywhere:  bash analysis/run_all.sh
# Needs Python 3.11+ with numpy, scipy, pandas and matplotlib.
set -u
A="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$A")"
SC="$ROOT/scaling"
PY="${PYTHON:-python3}"
# scb imports the openai client at module level; the analysis never calls a model, so a stand-in is enough when
# the package is not installed.
"$PY" -c "import openai" 2>/dev/null || export PYTHONPATH="$A/_stubs${PYTHONPATH:+:$PYTHONPATH}"
# scb (the benchmark) is imported from the repository root by every script.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$A" || exit 1
mkdir -p out/check
step() { echo "== $*"; }

step "ykazemi's title law and law with wrong guesses (scaling/scripts), written beside the committed *_aaai.json"
(cd "$SC" && "$PY" scripts/title_law.py results --extra ../wiki_game/data/swarm4round aaai aaai_or aaai_claude --out title_law_aaai_rerun.json)
(cd "$SC" && "$PY" scripts/law_wrong.py results --extra ../wiki_game/data/swarm4round aaai aaai_or aaai_claude --out law_wrong_aaai_rerun.json)

step "fuller.py";        "$PY" fuller.py
step "mixtures.py";      "$PY" mixtures.py
step "coverage.py";      "$PY" coverage.py
step "larger_n.py";      "$PY" larger_n.py
step "roles.py";         "$PY" roles.py
step "rule_models.py";   "$PY" rule_models.py
step "rule_models_tex.py (prints only)"; "$PY" rule_models_tex.py > out/rule_models_tex.txt
step "predict_mix16.py (registered forecast, recomputed)"; RECOMPUTE_INTO=out/check "$PY" predict_mix16.py
step "score_mix16.py";   "$PY" score_mix16.py
step "predict_mix3.py (registered forecast, recomputed)";  RECOMPUTE_INTO=out/check "$PY" predict_mix3.py "$SC"
step "score_mix3.py";    "$PY" score_mix3.py
step "fig14_points.py";  "$PY" fig14_points.py "$SC" > /dev/null
step "predict_pod5.py (registered forecast, recomputed)";  RECOMPUTE_INTO=out/check "$PY" predict_pod5.py
step "score_pod5.py";    "$PY" score_pod5.py
step "rounds.py";        "$PY" rounds.py "$SC/results" "$ROOT/wiki_game/data/swarm4round" "$SC/aaai" > rounds.md
step "late_holders.py";  "$PY" late_holders.py "$SC/results" "$ROOT/wiki_game/data/swarm4round" "$SC/aaai" "$ROOT/runs" > late_holders.md
step "law_points.py";    "$PY" law_points.py "$SC/results/law_wrong_aaai.json" "$SC/results/title_law_aaai.json" > law_points.md
step "byround.py";       "$PY" byround.py "$SC" > out/byround.txt   # writes byround.json
step "agreement.py";     "$PY" agreement.py "$SC" > out/check/agreement.md   # agreement.md adds a hand-written reading

step "compare with expected/"
# Registered forecasts are recomputed into out/check/; their only expected difference is the hackathon commit that
# predict_mix3.py names (it reads it from git).
for f in predictions_mix16 predictions_mix3 predictions_pod5; do
  cmp -s "$f.md" "out/check/$f.md" && echo "same     $f.md (recomputed)" || echo "DIFFERS  $f.md (recomputed, $(diff "$f.md" "out/check/$f.md" | grep -c '^[<>]') lines)"
done
for f in expected/*.md; do
  b=$(basename "$f")
  case "$b" in predictions_*|agreement.md) continue ;; esac
  # scaling_tests.md indexes the green text by hand; byround.md is byround.json's table, checked below
  [ "$b" = scaling_tests.md ] || [ "$b" = byround.md ] && continue
  if [ ! -f "$b" ]; then echo "MISSING  $b"; elif cmp -s "$f" "$b"; then echo "same     $b"; else echo "DIFFERS  $b ($(diff "$f" "$b" | grep -c '^[<>]') lines)"; fi
done
n=$(grep -c "" out/check/agreement.md)
head -n "$n" agreement.md | cmp -s - out/check/agreement.md && echo "same     agreement.md (all but its hand-written reading)" || echo "DIFFERS  agreement.md"
for f in title_law law_wrong; do
  "$PY" check_law_rows.py "$SC/results/${f}_aaai.json" "$SC/results/${f}_aaai_rerun.json"
done
"$PY" - <<'PYEOF'
import json, re
J = json.load(open("byround.json"))
rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in open("byround.md") if re.match(r"\| (Qwen|Gemma)", l)]
rows = [c for c in rows if len(c) == 9 and c[1].isdigit()]
bad = 0
for c in rows:
    r = [r for r in J if (c[0] in r["model"] or r["model"] in c[0]) and r["N"] == int(c[1])][0]
    bad += [f'{r[h][k]:.1f}' for h in ("r13", "r24") for k in ("obs", "pred")] != [c[3], c[4], c[6], c[7]]
print(("same    " if not bad else "DIFFERS ") + f" byround.md: {len(rows) - bad} of {len(rows)} rows match byround.json")
PYEOF

step "the hackathon repository's own outputs (scaling/scripts), rerun beside the committed files and compared"
# predict_swarm.json: the hackathon-version figures; posting.json, holders_seekers.json: README items 16-17;
# the rest: the green text (sources/scaling_tests.md). Each is rerun as scaling/README.md runs it.
rerun() { out=$1; shift; (cd "$SC" && "$PY" "$@" --out "${out%.json}_rerun.json" > /dev/null) || echo "FAILED   $out"; }
rerun predict_swarm.json   scripts/predict_swarm.py results --aware-base 4 --models qwen3-8b
rerun posting.json         scripts/analyze_posting.py results --extra ../wiki_game/data/swarm4round
rerun holders_seekers.json scripts/holders_seekers.py results
rerun plant.json           scripts/analyze_plant.py results
rerun ablations.json       scripts/analyze_ablations.py results
rerun twin.json            scripts/analyze_twin.py results
rerun transplant.json      scripts/analyze_transplant.py results --source aaai
rerun page_content.json    scripts/analyze_page_content.py results --source aaai
rerun twin_page.json       scripts/analyze_twin_page.py results --reference
for f in predict_swarm posting holders_seekers plant ablations twin transplant page_content twin_page; do
  "$PY" check_json.py "$SC/results/$f.json" "$SC/results/${f}_rerun.json"
done
