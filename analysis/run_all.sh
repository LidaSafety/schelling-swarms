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
(cd "$SC" && "$PY" scripts/title_law.py results --extra ../runs/swarm4round aaai aaai_or aaai_claude --out title_law_aaai_rerun.json)
(cd "$SC" && "$PY" scripts/law_wrong.py results --extra ../runs/swarm4round aaai aaai_or aaai_claude --out law_wrong_aaai_rerun.json)

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
step "rounds.py";        "$PY" rounds.py "$SC/results" "$ROOT/runs/swarm4round" "$SC/aaai" > rounds.md
step "late_holders.py";  "$PY" late_holders.py "$SC/results" "$ROOT/runs/swarm4round" "$SC/aaai" "$ROOT/runs" > late_holders.md
step "law_points.py";    "$PY" law_points.py "$SC/results/law_wrong_aaai.json" "$SC/results/title_law_aaai.json" > law_points.md
step "byround.py";       "$PY" byround.py "$SC" > out/byround.txt   # writes byround.json

step "compare with expected/"
# Registered forecasts are recomputed into out/check/; their only expected difference is the hackathon commit that
# predict_mix3.py names (it reads it from git).
for f in predictions_mix16 predictions_mix3 predictions_pod5; do
  cmp -s "$f.md" "out/check/$f.md" && echo "same     $f.md (recomputed)" || echo "DIFFERS  $f.md (recomputed, $(diff "$f.md" "out/check/$f.md" | grep -c '^[<>]') lines)"
done
for f in expected/*.md; do
  b=$(basename "$f")
  case "$b" in predictions_*) continue ;; esac
  # scaling_tests.md indexes the green text by hand; byround.md is byround.json's table, checked below
  [ "$b" = scaling_tests.md ] || [ "$b" = byround.md ] && continue
  if [ ! -f "$b" ]; then echo "MISSING  $b"; elif cmp -s "$f" "$b"; then echo "same     $b"; else echo "DIFFERS  $b ($(diff "$f" "$b" | grep -c '^[<>]') lines)"; fi
done
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
