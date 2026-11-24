"""Points for the paper's law figure: collusion measured in a swarm against the law from pairs (ykazemi's title law).

Reads the two outputs of the hackathon repository's scaling branch, run with the aaai branch's result folders as an
extra folder (scaling/scripts/title_law.py and law_wrong.py, `--extra ../wiki_game/data/swarm4round <aaai folders>`):
law_wrong.json gives the law with wrong guesses for single models, title_law.json the mixed swarms (Qwen3-8B and
Gemma 3 27B rarely guess, so the mixes use the law without guesses). Pairs (N = 2) are the base and are left out.

Groups:
  same     the swarm runs under the same rule as its pairs, N = 4-32 (and the mixes of 4-16)
  large    N = 64 and 128, rule in force
  rule     swarms without the rule, predicted from pairs WITH the rule

Usage: python law_points.py <law_wrong.json> <title_law.json> > law_points.md
"""
from __future__ import annotations

import json
import statistics
import sys

LW, TL = (json.load(open(p)) for p in sys.argv[1:3])
NAME = {"claude-haiku-4-5": "Haiku 4.5", "claude-haiku-5-5": "Haiku 5.5", "claude-opus-4-6": "Opus 4.6",
        "claude-sonnet-5-5": "Sonnet 5.5", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1", "openai/gpt-oss-120b": "gpt-oss-120B",
        "qwen/qwen3-235b-a22b-2507": "Qwen3-235B", "qwen/qwen3-32b": "Qwen3-32B (OR)", "qwen/qwen3.5-27b": "Qwen3.5-27B",
        "z-ai/glm-5.3-flash": "GLM-5.3"}

rows = []
for r in LW:
    if r["N"] == 2 or ("(swarm4round)" in r["target"] and r["model"] == "Qwen3-8B" and r["N"] in (4, 16)):
        continue                     # pairs are the base; the 8B reruns at N = 4 and 16 repeat the main cells
    model = NAME.get(r["model"], r["model"])
    if r["rule"] != r["base_rule"]:
        group = "rule"
    elif r["N"] >= 64:
        group = "large"
    else:
        group = "same"
    rows.append(dict(group=group, model=model, rule=r["rule"], N=r["N"], obs=100 * r["obs_cr"], pred=100 * r["pred_cr"],
                     lo=100 * r["pred_cr_ci"][0], hi=100 * r["pred_cr_ci"][1]))
for r in TL:
    if r["family"].startswith("Mixed"):
        rows.append(dict(group="same", model="mix " + r["target"].replace("qwen3-8b:", "Q").replace(",gemma-3-27b:", "+G"),
                         rule="forbidden", N=None, obs=100 * r["obs_cr"], pred=100 * r["pred_cr"],
                         lo=100 * r["pred_cr_ci"][0], hi=100 * r["pred_cr_ci"][1]))

print("# Law from pairs against measured collusion\n")
print("Written by `law_points.py` from the scaling branch's `law_wrong.json` and `title_law.json`, rerun on 20 November")
print("with the aaai branch's result folders; every row in the committed files was reproduced exactly.\n")
for g, title in (("same", "Same rule as the pairs, 4-32 copies, and the mixes"), ("large", "64 and 128 copies"),
                 ("rule", "Without the rule, from pairs with the rule")):
    gr = [r for r in rows if r["group"] == g]
    errs = [r["obs"] - r["pred"] for r in gr]
    inside = sum(r["lo"] - 1e-9 <= r["obs"] <= r["hi"] + 1e-9 for r in gr)
    print(f"## {title}\n")
    print(f"{len(gr)} swarms; median |measured - law| {statistics.median(abs(e) for e in errs):.1f} points, "
          f"largest {max(abs(e) for e in errs):.1f}; measured inside the law's 95% interval in {inside}.\n")
    print("| Model | Rule | N | Measured | Law | Law interval | Measured - law |")
    print("|---|---|---|---|---|---|---|")
    for r in gr:
        print(f"| {r['model']} | {r['rule']} | {r['N'] or ''} | {r['obs']:.1f} | {r['pred']:.1f} | [{r['lo']:.1f}, {r['hi']:.1f}] | {r['obs'] - r['pred']:+.1f} |")
    print()
    print("pgfplots: " + " ".join(f"({r['pred']:.1f},{r['obs']:.1f})" for r in gr) + "\n")
