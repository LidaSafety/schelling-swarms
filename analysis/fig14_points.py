"""Points of the paper's Figure 14 (fig:simplex): the 29 measured mixes of 16 copies (4+4+4: 12) of Qwen3-4B, Qwen3-8B
and Gemma 3 27B, i.e. the 22 of pod J (benchmark results/swarm_mix3) and the seven 8B + Gemma mixes of pod I
(results/swarm_mix16).

Left panel: one box per mix, solid up to the measured title match (pooled, from out/results_mix3.json and
out/results_mix16.json), light from there up to the weighted average of the pure swarms. Right panel: the collusion
rate against the prediction from pairs with wrong guesses, computed for every mix by the same code (the hackathon
repo's scaling/scripts/predict_mix3.py, run_chances on each measured run; for the 22 it equals the registered forecast).

Writes fig14_points.md, out/fig14_points.json and out/fig14_boxes.tex (the TikZ faces of the boxes, farthest first for
the figure's view, azimuth -20 or -15).

  PYTHONPATH=<hackathon clone> python fig14_points.py <hackathon clone>/scaling   (numpy and openai importable)
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES, RUNS  # noqa: E402
H = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SCALING
sys.path.insert(0, str(H / "scripts"))
sys.path.insert(0, str(H.parent))
import predict_mix3 as pm  # noqa: E402
from analyze_posting import run_counts  # noqa: E402
from predict_swarm import normalize  # noqa: E402
from scb.io import read_jsonl  # noqa: E402
from title_law import cells, condition, fixed, load  # noqa: E402

rows = load(H / "results", [RUNS / "swarm4round"])
base_rows = []
for m in pm.MODELS:
    cm = cells(rows, m, fixed())
    base_rows += cm.get(("results", 2)) or next((g for (s, n), g in cm.items() if n == 2), [])
base = pm.MixBase(base_rows)

R3 = {tuple(r["counts"]): r for r in json.loads((HERE / "out" / "results_mix3.json").read_text())["rows"]}
M16 = {t["swarm"]: t for t in json.loads((HERE / "out" / "results_mix16.json").read_text())["title"]}
mixes = []
for folder in ("swarm_mix3", "swarm_mix16"):
    for f in sorted((RES / folder / "raw").glob("*.jsonl.gz")):
        g = [r for r in (normalize(x) for x in read_jsonl(f) if not x.get("error")) if condition(r) == fixed()]
        ms = Counter(g[0]["seat_models"].values())
        a, b, c = ms["qwen3-4b"], ms["qwen3-8b"], ms["gemma-3-27b"]
        obs = sum((run_counts(r) for r in g), Counter())
        if folder == "swarm_mix3":
            t, w = R3[(a, b, c)]["pooled"], R3[(a, b, c)]["w"]
        else:
            tt = M16[f"{b}Q+{c}G"]
            t, w = tt["pooled"], tt["avg"]
        k = sum(v > 0 for v in (a, b, c))
        fam = "three" if k == 3 else "4B + 8B" if c == 0 else "4B + Gemma" if b == 0 else "8B + Gemma"
        mixes.append(dict(mix=f"{a}+{b}+{c}", a=a, b=b, c=c, N=a + b + c, games=len(g), family=fam, title=t, avg=w,
                          cr=100 * obs["used"] / obs["ch"], pred=100 * pm.predict(base, pm.run_chances(g))["cr"]))
assert len(mixes) == 29, len(mixes)

# TikZ faces: solid box to the measured title match, light (see-through, outlined) box from there to the weighted average
COL = {"three": "cB", "4B + 8B": "cC", "4B + Gemma": "cD", "8B + Gemma": "cE"}
hx = hy = 0.013


def poly(pts, style):
    return f"\\fill[{style}] " + " -- ".join(f"(axis cs:{x:.4f},{y:.4f},{z:.3f})" for x, y, z in pts) + " -- cycle;"


def box(X, Y, z0, z1, col, light):
    side = f"{col}!80!black, fill opacity=0.28, draw={col}!80!black, line width=0.2pt" if light else f"{col}!72!black"
    front = f"{col}, fill opacity=0.28, draw={col}!80!black, line width=0.2pt" if light else col
    x0, x1, y0, y1 = X - hx, X + hx, Y - hy, Y + hy
    out = [poly([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], side),     # left face, toward the viewer
           poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], front)]    # front face
    if light:
        out.append(poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                        f"{col}!40!white, draw={col}!80!black, line width=0.2pt"))  # top face
    return out


for m in mixes:
    m["X"], m["Y"] = (m["b"] + m["c"] / 2) / m["N"], 0.866 * m["c"] / m["N"]
order = sorted(mixes, key=lambda m: -(m["Y"] * math.cos(math.radians(20)) + m["X"] * math.sin(math.radians(20))))
tex = []
for m in order:
    tex.append(f"% {m['mix']}: measured {m['title']:.3f}, weighted average {m['avg']:.3f}")
    tex += box(m["X"], m["Y"], 0, m["title"], COL[m["family"]], False)
    tex += box(m["X"], m["Y"], m["title"], m["avg"], COL[m["family"]], True)
(HERE / "out" / "fig14_boxes.tex").write_text("\n".join(tex) + "\n")
(HERE / "out" / "fig14_points.json").write_text(json.dumps(mixes, indent=1))

lines = ["# Points of Figure 14 (fig:simplex)\n",
         "Written by `fig14_points.py`. 4B+8B+Gemma copies; title match pooled over round-1 titles; weighted average of "
         "the pure swarms' title match; CR measured and predicted from pairs with wrong guesses (each model's own pairs; "
         "the hackathon repo's `scaling/scripts/predict_mix3.py`, run on each measured run).\n",
         "| Mix | Family | N | Games | Title match | Weighted average | CR measured | CR predicted |",
         "|---|---|---|---|---|---|---|---|"]
for m in sorted(mixes, key=lambda m: (m["family"], -m["a"], -m["b"])):
    lines.append(f"| {m['mix']} | {m['family']} | {m['N']} | {m['games']} | {m['title']:.3f} | {m['avg']:.3f} | "
                 f"{m['cr']:.1f} | {m['pred']:.1f} |")
lines.append("\npgfplots, right panel (predicted, measured):")
for fam in COL:
    lines.append(f"- {fam}: " + " ".join(f"({m['pred']:.1f},{m['cr']:.1f})" for m in order if m["family"] == fam))
(HERE / "fig14_points.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
