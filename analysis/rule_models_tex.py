"""The LaTeX for the paper's rule-across-models figure and appendix table, from out/rule_models.json (rule_models.py),
and the rank correlations the text quotes. Prints; nothing is written. Rerun after rule_models.py.
  python rule_models_tex.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = [r for r in json.loads((HERE / "out" / "rule_models.json").read_text())
        if not r["how"].startswith("OpenRouter, DeepInfra fp8") or r["model"] != "Qwen3-32B"]   # our own 32B run only
rows = [r for r in rows if r["accuracy"] == r["accuracy"]]                                       # with a quiz score
SHORT = {"Qwen3-4B": "4B", "Qwen3-8B": "8B", "Qwen3-14B": "14B", "Qwen3-32B": "32B", "Gemma 3 27B": "Gemma 3",
         "Qwen3-235B-A22B-2507": "Qwen3-235B", "gpt-oss-120B (low reasoning)": "gpt-oss", "GLM-5.3 Flash (low reasoning)": "GLM-5.3",
         "DeepSeek V4.1 Flash": "DeepSeek", "Qwen3.5-27B": "Qwen3.5", "Claude Haiku 4.5": "Haiku 4.5",
         "Claude Haiku 5.5": "Haiku 5.5", "Claude Sonnet 5.5": "Sonnet 5.5", "Claude Opus 4.6": "Opus 4.6"}
COLOR = lambda m: ("cB" if m.startswith("Claude") else "cE" if m.startswith("Gemma") else  # noqa: E731
                   "cA" if m.startswith("Qwen") else "cC")


def rank(x):
    o = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in set(x):                       # average ranks for ties
        idx = [i for i, y in enumerate(x) if y == v]
        o[idx] = np.mean(o[idx])
    return o


def spearman(a, b):
    return float(np.corrcoef(rank(list(a)), rank(list(b)))[0, 1])


acc = [r["accuracy"] for r in rows]
for k, name in (("cr_rule", "CR with the rule"), ("cr_norule", "CR without the rule"), ("diff", "difference")):
    print(f"% Spearman, quiz accuracy vs {name}: {spearman(acc, [r[k] for r in rows]):+.2f} (n = {len(rows)})")
print()
print("% --- figure: one segment per model, filled = rule, open = no rule")
for r in sorted(rows, key=lambda r: r["accuracy"]):
    c, x, y1, y0 = COLOR(r["model"]), r["accuracy"], r["cr_rule"], r["cr_norule"]
    print(f"\\draw[{c}, line width=0.7pt] (axis cs:{x:.1f},{y1:.1f}) -- (axis cs:{x:.1f},{y0:.1f});"
          f"  % {r['model']}")
    print(f"\\addplot[only marks, mark=*, mark size=1.8pt, {c}, forget plot] coordinates {{({x:.1f},{y1:.1f})}};")
    print(f"\\addplot[only marks, mark=o, mark size=1.8pt, {c}, mark options={{fill=white}}, forget plot] "
          f"coordinates {{({x:.1f},{y0:.1f})}};")
    top = max(y1, y0)
    if top < 2:      # a model at 0 either way: label below the axis line (the plot starts at -8)
        anchor = "north east" if r["model"] == "Claude Haiku 5.5" else "north"
        print(f"\\node[font=\\tiny, text={c}, anchor={anchor}] at (axis cs:{x:.1f},-1) {{{SHORT[r['model']]}}};")
    else:
        anchor = "south east" if r["model"].startswith("GLM") else "south"     # clear of DeepSeek's label
        print(f"\\node[font=\\tiny, text={c}, anchor={anchor}] at (axis cs:{x:.1f},{top:.1f}) {{{SHORT[r['model']]}}};")
print()
print("% --- appendix table rows")
for r in sorted(json.loads((HERE / "out" / "rule_models.json").read_text()), key=lambda r: r["accuracy"] if r["accuracy"] == r["accuracy"] else 0):
    a = "--" if r["accuracy"] != r["accuracy"] else f"{r['accuracy']:.1f}"
    d = r["ci_diff"]
    print(f"{r['model']} & {r['how']} & {a} & {r['games'][0]} & {r['cr_rule']:.1f} & {r['cr_norule']:.1f} & "
          f"${r['diff']:+.1f}$ [${d[0]:+.1f}$, ${d[1]:+.1f}$] \\\\")
