#!/usr/bin/env python3
"""Paired 95% intervals for the differences in the paper's table "The rule across models" (tab:rulemodels).

  python analysis/rule_models_paired.py > analysis/rule_models_paired.md   (from the repository root)

Both arms of each row play the same quizzes, so the interval resamples seeds and keeps the two arms of a seed together,
with the same bootstrap as rule_row.py (tab:rule): 5,000 resamples, random.Random(0), 2.5th and 97.5th percentiles.
CR is chances used over chances, pooled over the resampled games.
"""
import gzip, json, random

from paths import RES, S4R

B = ""
ROWS = [  # (row, rule-arm file, no-rule-arm file); 16 copies, 2 answers per copy, nothing planted
    ("Qwen3-4B", B + "swarm_ref_4b/raw/qwen3-4b.jsonl.gz", B + "swarm_ruleoff2/raw/qwen3-4b.jsonl.gz"),
    ("Qwen3-8B", B + "swarm_ref_8b/raw/qwen3-8b.jsonl.gz", B + "swarm_ruleoff2/raw/qwen3-8b.jsonl.gz"),
    ("Qwen3-14B", B + "swarm_ref_14b/raw/qwen3-14b.jsonl.gz", B + "swarm_ruleoff/raw/qwen3-14b.jsonl.gz"),
    ("Gemma 3 27B", B + "swarm_gemma_ladder_n16/raw/gemma-3-27b.jsonl.gz", B + "swarm_gemma_ruleoff_n16/raw/gemma-3-27b.jsonl.gz"),
    ("Qwen3-32B (OpenRouter)", B + "swarm_or_qwen_qwen3-32b/raw/qwen_qwen3-32b.jsonl.gz", None),
    ("Qwen3-32B (our GPU)", B + "swarm_ref_32b/raw/qwen3-32b-bf16.jsonl.gz", B + "swarm_ruleoff/raw/qwen3-32b-bf16.jsonl.gz"),
    ("gpt-oss-120B", B + "swarm_or_openai_gpt-oss-120b/raw/openai_gpt-oss-120b.jsonl.gz", None),
    ("Qwen3-235B-A22B-2507", B + "swarm_or_qwen_qwen3-235b-a22b-2507/raw/qwen_qwen3-235b-a22b-2507.jsonl.gz", None),
    ("Qwen3.5-27B", B + "swarm_or_qwen_qwen3_5-27b/raw/qwen_qwen3.5-27b.jsonl.gz", None),
    ("Claude Haiku 4.5", B + "swarm_claude_haiku45/raw/claude-haiku-4-5.jsonl.gz", None),
    ("GLM-5.3 Flash", B + "swarm_or_z-ai_glm-5_3-flash/raw/z-ai_glm-5.3-flash.jsonl.gz", None),
    ("Claude Haiku 5.5", B + "swarm_claude_haiku/raw/claude-haiku-5-5.jsonl.gz", None),
    ("DeepSeek V4.1 Flash", B + "swarm_or_deepseek_deepseek-v4_1-flash/raw/deepseek_deepseek-v4.1-flash.jsonl.gz", None),
    ("Claude Sonnet 5.5", B + "swarm_claude_sonnet55/raw/claude-sonnet-5-5.jsonl.gz", None),
    ("Claude Opus 4.6", B + "swarm_claude_opus46/raw/claude-opus-4-6.jsonl.gz", None),
]


def resolve(p):
    folder, rest = p.split("/", 1)
    return (S4R / folder if (S4R / folder).exists() else RES / folder) / rest


def load(path, rule):
    """seed -> (chances used, chances) for the 16-copy games of one arm with nothing planted."""
    out = {}
    for r in map(json.loads, gzip.open(resolve(path), "rt")):
        n_agents = r.get("n_agents") or len(r.get("holds", {}))
        if n_agents != 16 or r.get("rule") != rule or str(r.get("plant")) not in ("None", "none"):
            continue
        ch = r["chances"]
        out[r["seed"]] = (ch["used"], ch["n"]) if isinstance(ch, dict) else (r["n_used"], ch)
    return out


def cr(arm, seeds):
    return 100 * sum(arm[s][0] for s in seeds) / sum(arm[s][1] for s in seeds)


if __name__ == "__main__":
    for name, f_rule, f_off in ROWS:
        rule, off = load(f_rule, "forbidden"), load(f_off or f_rule, "silent")
        seeds = sorted(set(rule) & set(off))
        rng = random.Random(0)
        diffs = sorted(cr(off, ss) - cr(rule, ss) for ss in ([rng.choice(seeds) for _ in seeds] for _ in range(5000)))
        print(f"{name:24s} games {len(seeds):3d}  CR {cr(rule, seeds):5.1f} -> {cr(off, seeds):5.1f}  "
              f"difference {cr(off, seeds) - cr(rule, seeds):+5.1f} [{diffs[125]:+.1f}, {diffs[4874]:+.1f}]")
