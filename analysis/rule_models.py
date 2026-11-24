"""The rule ablation at N = 16 across models: CR with and without the rule sentence, for the Qwen3 sizes (our runs, as
in the paper's Table 9) and for the models run through APIs on 19 November (Claude through the Anthropic API, the rest
through OpenRouter; benchmark branch claude-api, entries 22 and 23 of configs/experiments.yaml).

Every arm is the same games (seed 1, games 10000-), 2 answers per copy, unaware, no listing, no log. CR is uses over
chances; intervals resample whole games (B = 2000), and the difference resamples the two arms independently (the arms
share seeds, so this is conservative). Also per arm: share of write steps with an edit, share of read steps with a view,
and share of seekers' answers left empty. Writes rule_models.md and out/rule_models.json. Rerun with
`python rule_models.py`; a run not yet on disk is skipped.
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES  # noqa: E402
B = 2000
rng = np.random.default_rng(20261119)

# label, how it is run, folder with the rule, folder without (both arms in one folder for the API models), and the
# model's name in the capability quiz (benchmark results/capability_mmlupro*, scb capability: 504 MMLU-Pro questions)
MODELS = [
    ("Qwen3-4B", "vLLM, bf16", RES / "swarm4round/swarm_ref_4b", RES / "swarm_ruleoff2", "qwen3-4b"),
    ("Qwen3-8B", "vLLM, bf16", RES / "swarm4round/swarm_ref_8b", RES / "swarm_ruleoff2", "qwen3-8b"),
    ("Qwen3-14B", "vLLM, bf16", RES / "swarm4round/swarm_ref_14b", RES / "swarm_ruleoff", "qwen3-14b"),
    ("Qwen3-32B", "vLLM, bf16", RES / "swarm4round/swarm_ref_32b", RES / "swarm_ruleoff", "qwen3-32b-bf16"),
    ("Gemma 3 27B", "vLLM, bf16", RES / "swarm_gemma_ladder_n16", RES / "swarm_gemma_ruleoff_n16", "gemma-3-27b"),
    ("Qwen3-32B", "OpenRouter, DeepInfra fp8", RES / "swarm_or_qwen_qwen3-32b", None, "qwen/qwen3-32b"),
    ("Qwen3-235B-A22B-2507", "OpenRouter, DeepInfra fp8", RES / "swarm_or_qwen_qwen3-235b-a22b-2507", None, "qwen/qwen3-235b-a22b-2507"),
    # Gemma 4 31B: piloted at 0 -> 0; the full run was rate-limited by its host and then stopped (19 November)
    ("gpt-oss-120B (low reasoning)", "OpenRouter, Crusoe bf16", RES / "swarm_or_openai_gpt-oss-120b", None, "openai/gpt-oss-120b"),
    ("GLM-5.3 Flash (low reasoning)", "OpenRouter, Parasail fp8", RES / "swarm_or_z-ai_glm-5_3-flash", None, "z-ai/glm-5.3-flash"),
    ("DeepSeek V4.1 Flash", "OpenRouter, DeepInfra fp8", RES / "swarm_or_deepseek_deepseek-v4_1-flash", None, "deepseek/deepseek-v4.1-flash"),
    ("Qwen3.5-27B", "OpenRouter, DeepInfra fp8", RES / "swarm_or_qwen_qwen3_5-27b", None, "qwen/qwen3.5-27b"),
    ("Claude Haiku 4.5", "Anthropic API", RES / "swarm_claude_haiku45", None, "claude-haiku-4-5"),
    ("Claude Haiku 5.5", "Anthropic API", RES / "swarm_claude_haiku", None, "claude-haiku-5-5"),
    ("Claude Sonnet 5.5", "Anthropic API", RES / "swarm_claude_sonnet55", None, "claude-sonnet-5-5"),
    ("Claude Opus 4.6", "Anthropic API, 50 games per arm", RES / "swarm_claude_opus46", None, "claude-opus-4-6"),
]
CAP = {}
for d in ("capability_mmlupro", "capability_mmlupro_pod"):
    f = RES / d / "summary.json"
    if f.exists():
        CAP.update(json.loads(f.read_text())["models"])
MODEL_KEY = {"Gemma 3 27B": "gemma-3-27b", "Qwen3-4B": "qwen3-4b", "Qwen3-8B": "qwen3-8b", "Qwen3-14B": "qwen3-14b", "Qwen3-32B": "qwen3-32b-bf16"}


def games(folder: Path, rule: str, model_key: str | None):
    out = []
    for f in sorted((folder / "raw").glob("*.jsonl*")):
        if model_key and not f.name.startswith(model_key + "."):
            continue
        with (gzip.open if f.suffix == ".gz" else open)(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if ("error" in r or r.get("rule") != rule or r.get("n_agents") != 16 or r.get("f_inv") != 2
                        or r.get("awareness") != "unaware" or r.get("plant") not in (None, "none")):
                    continue
                c = defaultdict(int)
                for a, ag in r["agents"].items():
                    for x in ag["answers"]:
                        if x["chance"]:
                            c["ch"] += 1
                            c["use"] += bool(x["used_chosen"])
                            c["empty"] += x["answer"] is None
                    ev = r["events"][a]      # lists only the steps that edited or viewed a page
                    rounds = r.get("n_questions") or 4
                    c["w"] += sum(1 for w in ev["writes"] if w.get("path"))
                    c["ws"] += rounds        # one write step and one read step per copy per round
                    c["r"] += sum(1 for x in ev["reads"] if x.get("path"))
                    c["rs"] += rounds
                out.append(c)
    return out


def cr(G):
    ch, use = np.array([g["ch"] for g in G], float), np.array([g["use"] for g in G], float)
    idx = rng.integers(0, len(G), (B, len(G)))
    return 100 * use.sum() / ch.sum(), 100 * use[idx].sum(1) / ch[idx].sum(1)


lines, J = [], []
say = lines.append
say("# The rule ablation across models at N = 16\n")
say("Quiz accuracy: % correct on the capability quiz (504 MMLU-Pro questions, each model with its game settings; a reply with no answer within 2048 tokens counts as wrong). CR in % with 95% intervals from resampling games; the difference is no rule less rule. Writes, reads: share of "
    "write and read steps that edit or view a page. Empty: share of seekers' answers left empty. Rerun with "
    "`python rule_models.py`.\n")
say("| Model | Run on | Quiz accuracy | Games per arm | CR, rule | CR, no rule | Difference | Writes, rule → no rule | "
    "Reads, rule → no rule | Empty answers, rule → no rule |")
say("|---|---|---|---|---|---|---|---|---|---|")
for label, how, on, off, capname in MODELS:
    key = MODEL_KEY.get(label) if how.startswith("vLLM") else None
    g_on, g_off = games(on, "forbidden", key), games(off or on, "silent", key)
    if not g_on or not g_off:
        continue
    (p1, b1), (p0, b0) = cr(g_on), cr(g_off)
    d = b0 - b1
    s = lambda G, a, b: 100 * sum(g[a] for g in G) / max(sum(g[b] for g in G), 1)  # noqa: E731
    c = CAP.get(capname)
    acc = (100 * c["accuracy"], [100 * x for x in c["ci95"]], c["unparsed"]) if c else (float("nan"), [float("nan")] * 2, 0)
    row = dict(model=label, how=how, accuracy=acc[0], accuracy_ci=acc[1], unparsed=acc[2], games=[len(g_on), len(g_off)], cr_rule=p1, ci_rule=list(np.percentile(b1, [2.5, 97.5])),
               cr_norule=p0, ci_norule=list(np.percentile(b0, [2.5, 97.5])), diff=p0 - p1,
               ci_diff=list(np.percentile(d, [2.5, 97.5])), writes=[s(g_on, "w", "ws"), s(g_off, "w", "ws")],
               reads=[s(g_on, "r", "rs"), s(g_off, "r", "rs")], empty=[s(g_on, "empty", "ch"), s(g_off, "empty", "ch")])
    J.append(row)
    say(f"| {label} | {how} | {acc[0]:.1f} [{acc[1][0]:.1f}, {acc[1][1]:.1f}] | {len(g_on)}, {len(g_off)} | {p1:.1f} [{row['ci_rule'][0]:.1f}, {row['ci_rule'][1]:.1f}] | "
        f"{p0:.1f} [{row['ci_norule'][0]:.1f}, {row['ci_norule'][1]:.1f}] | {p0 - p1:+.1f} [{row['ci_diff'][0]:+.1f}, "
        f"{row['ci_diff'][1]:+.1f}] | {row['writes'][0]:.0f} → {row['writes'][1]:.0f} | {row['reads'][0]:.0f} → "
        f"{row['reads'][1]:.0f} | {row['empty'][0]:.0f} → {row['empty'][1]:.0f} |")
(HERE / "rule_models.md").write_text("\n".join(lines) + "\n")
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "rule_models.json").write_text(json.dumps(J, indent=1, default=float))
print("\n".join(lines))
