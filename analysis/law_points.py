#!/usr/bin/env python3
"""Measured collusion against the prediction from pairs (the law), as rows: law_points.md.

  cd aaai-paper/sources && uv run --no-project python law_points.py ../../Swarm_schelling_hackathon/scaling

Reads two outputs of the hackathon repository's scaling scripts, unchanged:
  results/law_wrong_aaai.json   every single-model swarm (law_wrong.py with the aaai folders; see scaling/README.md)
  results/title_law_aaai.json   the 11 mixes of Qwen3-8B and Gemma 3 27B (title_law.py; law_wrong.py skips mixes)
and writes law_points.md next to this file. Behind Figure 7 (fig:law), tab:lawsizes, tab:law and the medians in
sec:law and app:law.

Sections (rows in the order of the JSON files):
  1. Same rule as the pairs, N = 4-32, and the mixes: law_wrong rows whose swarm and pairs share the rule, then the
     mixes. When a model has the same N in two folders (Qwen3-8B at 4 and 16 copies: results and swarm4round), the
     first one is kept, which is the results folder.
  2. 64 and 128 copies, rule in force.
  3. Without the rule, predicted from pairs with the rule (the cross-test).
Medians, largest differences and the in-interval counts use the unrounded numbers; the tables print them to 0.1.

Use rate: law_wrong_aaai.json must come from law_wrong.py at or after commit e96acca, where a model whose pairs
contain no found value gets the assumed use rate u = 1 (--no-finds-uptake, default 1) instead of 0. Only Qwen3-32B
with the rule moves. The script stops if its 32B rule rows still have u = 0.
"""
import json, statistics, sys
from pathlib import Path

sc = Path(sys.argv[1]).resolve()
lw = json.load(open(sc / "results/law_wrong_aaai.json"))
tl = json.load(open(sc / "results/title_law_aaai.json"))
NAME = {"claude-haiku-4-5": "Haiku 4.5", "claude-haiku-5-5": "Haiku 5.5", "claude-opus-4-6": "Opus 4.6",
        "claude-sonnet-5-5": "Sonnet 5.5", "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1",
        "openai/gpt-oss-120b": "gpt-oss-120B", "qwen/qwen3-235b-a22b-2507": "Qwen3-235B",
        "qwen/qwen3-32b": "Qwen3-32B (OR)", "qwen/qwen3.5-27b": "Qwen3.5-27B", "z-ai/glm-5.3-flash": "GLM-5.3"}
assert all(x["uptake_base"] == 1.0 for x in lw if x["model"] == "Qwen3-32B" and x["base_rule"] == "forbidden"), \
    "law_wrong_aaai.json predates e96acca (Qwen3-32B with the rule has use rate 0): rerun law_wrong.py"


def row(model, rule, n, meas, law, ci):
    return dict(model=model, rule=rule, N=n, meas=100 * meas, law=100 * law, lo=100 * ci[0], hi=100 * ci[1])


def single(keep):
    """law_wrong rows passing keep, one per (model, rule, pairs' rule, N): the first in the file."""
    out, seen = [], set()
    for x in lw:
        # DeepSeek V4.1 Flash with 32 and 64 copies is scored against its own registered forecast (app:deepseek),
        # not in these groups; law_wrong.py includes it once swarm_deepseek_large is in results/.
        if x["model"] == "deepseek/deepseek-v4.1-flash" and x["N"] > 16:
            continue
        k = (x["model"], x["rule"], x["base_rule"], x["N"])
        if keep(x) and k not in seen:
            seen.add(k)
            out.append(row(NAME.get(x["model"], x["model"]), x["rule"], x["N"], x["obs_cr"], x["pred_cr"],
                           x["pred_cr_ci"]))
    return out


def mix_label(target):   # "qwen3-8b:10,gemma-3-27b:6" -> "mix Q10+G6"
    q, g = (s.split(":")[1] for s in target.split(","))
    return f"mix Q{q}+G{g}"


same = single(lambda x: x["rule"] == x["base_rule"] and 4 <= x["N"] <= 32)
mixes = [row(mix_label(x["target"]), "forbidden", None, x["obs_cr"], x["pred_cr"], x["pred_cr_ci"])
         for x in tl if x["family"].startswith("Mixed")]          # title_law.py runs the mixes with the rule (fixed())
big = single(lambda x: x["rule"] == x["base_rule"] == "forbidden" and x["N"] in (64, 128))
cross = single(lambda x: x["rule"] == "silent" and x["base_rule"] == "forbidden" and x["N"] >= 4)


def section(title, rows):
    d = [abs(r["meas"] - r["law"]) for r in rows]
    inside = sum(r["lo"] <= r["meas"] <= r["hi"] for r in rows)
    lines = [f"## {title}", "",
             f"{len(rows)} swarms; median |measured - law| {statistics.median(d):.1f} points, largest {max(d):.1f}; "
             f"measured inside the law's 95% interval in {inside}.", "",
             "| Model | Rule | N | Measured | Law | Law interval | Measured - law |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['rule']} | {'' if r['N'] is None else r['N']} | {r['meas']:.1f} | {r['law']:.1f} "
              f"| [{r['lo']:.1f}, {r['hi']:.1f}] | {r['meas'] - r['law']:+.1f} |" for r in rows]
    lines += ["", "pgfplots: " + " ".join(f"({r['law']:.1f},{r['meas']:.1f})" for r in rows), ""]
    return lines


md = ["# Law from pairs against measured collusion", "",
      "Written by `law_points.py` from the scaling branch's `law_wrong_aaai.json` and `title_law_aaai.json` (the 11",
      "mixes), with `law_wrong_aaai.json` rerun on 24 November with `law_wrong.py` after commit e96acca.",
      "Use rate: when a model's pairs contain no found value, its use rate $\\hat u$ is 0 of 0, unmeasured, and the law",
      "assumes $\\hat u = 1$ (`--no-finds-uptake 1`, the default). This moves only Qwen3-32B with the rule (0 finds in",
      "100 pairs); the Claude models with no finds have a predicted find rate of 0, so their law stays 0.", ""]
md += section("Same rule as the pairs, 4-32 copies, and the mixes", same + mixes)
md += section("64 and 128 copies", big)
md += section("Without the rule, from pairs with the rule", cross)
Path(__file__).with_name("law_points.md").write_text("\n".join(md) + "\n")
print(f"wrote {Path(__file__).with_name('law_points.md')}: {len(same + mixes)}, {len(big)}, {len(cross)} rows")
