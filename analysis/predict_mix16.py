"""Forecasts for the mixed swarms at N = 16, written before they run. Writes predictions_mix16.md and refuses to overwrite
it: the registered version is the one committed before the runs.

Runs (benchmark branch mixed-n16, configs/experiments.yaml entry 20; results/logs/podH/pod/mix16.sh): Gemma 3 27B alone
at N = 16, and Gemma shares in eighths: 14 + 2, 12 + 4, ..., 2 + 14 copies of Qwen3-8B + Gemma 3 27B; tag mix16,
seed 1, 100 games each, unaware, rule in force, no listing, no log. (First registered on 18 November with three mixes,
4 + 12, 8 + 8 and 12 + 4; replaced the same day, before any run, when the grid became eighths.)

1. Title match, from the paper's Appendix F (one round-1 title distribution per model, counted on pure swarms of
   N <= 8; read from out/mixtures.json, written by mixtures.py): f(x) for any two round-1 titles of the mix and
   f_N(x) = f(x) - gap / (N - 1) for two copies of one game.
2. Collusion: the paper's Section 4.3 says a mix colludes less than the weighted average of its pure swarms.
3. For the record, the tabled fuller model's forecasts for the same swarms (fuller.md section 6, drafted on
   17 November before any of these runs existed).
"""
import os
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# RECOMPUTE_INTO=<dir> writes the forecast there instead, to check it against the registered version.
OUT = Path(os.environ.get("RECOMPUTE_INTO", HERE)) / "predictions_mix16.md"
if OUT.exists():
    raise SystemExit(f"{OUT.name} exists; it is the registered version. Move it away to recompute.")

J = json.loads((HERE / "out" / "mixtures.json").read_text())
i, j = J["models"].index("8B"), J["models"].index("Gemma")
M = J["M"]
QQ, QG, GG = M[i][i], M[i][j], M[j][j]
D = QQ + GG - 2 * QG
N = 16
MIXES = [(14, 2), (12, 4), (10, 6), (8, 8), (6, 10), (4, 12), (2, 14)]

fuller = {}
sec6 = (HERE / "fuller.md").read_text().split("## 6.")[1].split("## 7.")[0]
for line in sec6.splitlines():
    m = re.match(r"\| (\d+) Qwen3-8B \+ (\d+) Gemma 3 27B \| 16 \| ([\d.]+) \| ([\d.]+) \| (-?[\d.]+) \|", line)
    if m:
        fuller[(int(m[1]), int(m[2]))] = dict(cr=float(m[3]), avg=float(m[4]), sag=-float(m[5]))
    m = re.match(r"\| Gemma 3 27B \| 16 \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
    if m:
        fuller["gemma"] = dict(simple=float(m[1]), fuller=float(m[4]))
assert len(fuller) == 4, fuller

lines = []
say = lines.append
say("# Forecasts for the mixed swarms at N = 16\n")
say("Written and committed before these runs start. Runs: Gemma 3 27B alone at N = 16, and Gemma shares in eighths, "
    "14 + 2, 12 + 4, ..., 2 + 14 copies of Qwen3-8B + Gemma 3 27B (benchmark branch `mixed-n16`, `configs/experiments.yaml` entry 20, run from "
    "`results/logs/podH/pod/mix16.sh`): tag `mix16`, seed 1 (games 10000–10099), 100 games each, unaware, rule in "
    "force, no listing, no log, 2 answers per copy, each model holding both halves of the answers. Qwen3-8B alone at "
    "N = 16 is the paper's main run (58.4%). Rerun with `python predict_mix16.py` (it will not overwrite this file). "
    "A first version with three mixes (Gemma shares 1/4, 1/2, 3/4) was committed earlier on 18 November and replaced "
    "before any run; its forecasts for those three mixes are unchanged here.\n")

say("## 1. Title match (the paper's Appendix F)\n")
say(f"One round-1 title distribution per model, counted on pure swarms of N ≤ 8 (`mixtures.md`): M_QQ = {QQ:.3f}, "
    f"M_QG = {QG:.3f}, M_GG = {GG:.3f}, D = {D:.3f}. For Gemma share x, f(x) = (1 − x)² M_QQ + 2x(1 − x) M_QG + "
    f"x² M_GG, the gap below the weighted average is x(1 − x) D, and two different copies of one game of N copies "
    f"match with f_N(x) = f(x) − x(1 − x) D / (N − 1). Measured as in `mixtures.py` section 2: *pooled*, any two "
    "round-1 titles of the condition (a title not paired with itself); *same game*, the round-1 titles of two "
    "different copies of one game.\n")
say("| Swarm | Gemma share x | f(x), pooled | f_N(x), same game | Weighted average |")
say("|---|---|---|---|---|")
rows = []
for nq, ng in MIXES:
    x = ng / N
    f = (1 - x) ** 2 * QQ + 2 * x * (1 - x) * QG + x ** 2 * GG
    fN = f - x * (1 - x) * D / (N - 1)
    rows.append((nq, ng, x, f, fN))
    say(f"| {nq}Q+{ng}G | {x:g} | **{f:.3f}** | **{fN:.3f}** | {(1 - x) * QQ + x * GG:.3f} |")
say(f"| Gemma alone | 1 | **{GG:.3f}** | **{GG:.3f}** | {GG:.3f} |")
say("")
say(f"**Registered:** every one of the {2 * len(MIXES) + 2} measured title matches above (pooled and same game, "
    f"{len(MIXES)} mixes and Gemma alone) is within 0.02 of its forecast. At N ≤ 8 the largest difference was 0.015 (`mixtures.md` section 2).\n")

say("## 2. Collusion below the weighted average (the paper's Section 4.3)\n")
say("The weighted average at N = 16 is (1 − x) × 58.4 + x × Gemma's measured CR at N = 16 from this session. Sag: "
    "weighted average less the mix's CR, with a 95% interval from resampling games (the three runs resampled "
    "independently, as in `mixtures.md`).\n")
say("**Registered:** the measured sag is positive in at least 6 of the 7 mixes, and its 95% interval lies above "
    "zero in at least 4 of them. The two mixes with only 2 copies of one model (14 + 2, 2 + 14) sit closest to a "
    "pure swarm, where the sag is smallest.\n")

say("## 3. For the record: the tabled fuller model\n")
say("The model of the paper's second version, which the AAAI version does not use. Its forecasts were drafted in "
    "`fuller.md` section 6 on 17 November, for three of the seven mixes only; they are copied here unchanged. "
    "Collusion rate in %.\n")
say("| Swarm | Fuller | Weighted average of its pure forecasts | Sag |")
say("|---|---|---|---|")
for nq, ng in MIXES:
    r = fuller.get((nq, ng))
    if r is None:
        continue
    say(f"| {nq}Q+{ng}G | {r['cr']:.1f} | {r['avg']:.1f} | {r['sag']:.1f} |")
say(f"| Gemma alone | {fuller['gemma']['fuller']:.1f} (simple model {fuller['gemma']['simple']:.1f}) | | |")
say("")
say("Two checks, recorded but not counted as a test of the paper: (a) each forecast within 8 points of the measured "
    "CR (the accuracy bound of `predictions_pod5.md`); (b) the measured sag at N = 16 is smaller than the measured "
    "sag at N = 8 for the same share (9.5 for x = 1/4, 10.7 for x = 1/2, 6.2 for x = 3/4), as the model forecasts "
    "(4–6 points).\n")
say("Scoring: a `score_mix16.py` is to be written after the runs, reading `results/swarm_mix16/`.")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
