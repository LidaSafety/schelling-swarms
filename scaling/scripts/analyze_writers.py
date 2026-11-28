#!/usr/bin/env python3
"""The writers test: is Gemma's text more influential for both readers, or does each reader respond to its own model's?

  .venv/bin/python scripts/analyze_writers.py results
      writes results/writers.json and prints, per reader and writer, how often late holders post their round-2 answer,
      and the tests registered in results/predictions_writers.md

Pairs of Gemma 3 27B (tag plant_writer_gemma) and of Qwen3-8B (plant_writer_q8) on the fixed prompt. Before round 1,
nothing or the model's answer page from item 26 rewritten note by note by Gemma 3 27B, Qwen3-8B or Llama 3.3 70B
(scripts/make_paraphrase_pages.py) sits on the pair's usual round-1 title for question 1. Late holders, treated and
posting as in scripts/analyze_page_content.py. Intervals resample seeds, all games of a seed, in every arm of both
readers, moving together. Reported: the same arms pooled with item 28's (the plant_style tags, other sample offsets),
the rewritten pages' lengths and bare-number shares, and the authorship check (results/writers_authorship.json).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_page_content import late_record  # noqa: E402
from analyze_transplant import Arms, rows_of  # noqa: E402
from title_law import condition, fixed, pct  # noqa: E402

MODELS = {"gemma-3-27b": ("plant_writer_gemma", "plant_style_gemma", "gemma", "{state} Construction Employment Share"),
          "qwen3-8b": ("plant_writer_q8", "plant_style_q8", "q8", "{state} construction occupation share")}
ARMS = ("nothing", "by_gemma", "by_q8", "by_llama")
# registered: label -> (reader, arm, arm it is compared with, sides, the test)
REGISTERED = {
    "a": ("gemma-3-27b", "by_gemma", "by_llama", 1, "Gemma reads: Gemma's rewrite above Llama's"),
    "b": ("qwen3-8b", "by_llama", "by_gemma", 1, "Qwen3-8B reads: Gemma's rewrite below Llama's"),
    "c": ("gemma-3-27b", "by_gemma", "by_q8", 1, "Gemma reads: Gemma's rewrite above Qwen3-8B's (item 28 again)"),
    "d": ("qwen3-8b", "by_q8", "by_gemma", 1, "Qwen3-8B reads: its own rewrite above Gemma's (item 28 again)"),
}
REPORTED = [("by_gemma", "nothing"), ("by_q8", "nothing"), ("by_llama", "nothing"), ("by_q8", "by_llama")]
BARE = re.compile(r"^\s*([A-Z][A-Za-z]*( [A-Z][A-Za-z]*)*\s*:\s*)?\d+(\.\d+)?\s*%?\.?\s*$")


def arm(r: dict, short: str) -> str:
    p = r.get("plant")
    f = p["file"] if isinstance(p, dict) and "file" in p else None
    if f is None:
        return "nothing"
    if f == f"{short}_by_{short}":
        return f"by_{short}"
    return f.removeprefix(f"{short}_") if f.startswith(f"{short}_by_") else {f"{short}_answers_q1": "own"}.get(f, f)


def collect(folder: Path, model: str, short: str, usual: str, item28: bool = False) -> dict:
    recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not folder.is_dir():
        return recs
    for r in rows_of(folder):
        if r["model"] != model or r["n_agents"] != 2 or condition(r) != fixed():
            continue
        p = r.get("plant")
        x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
        if x:
            recs[arm(r, short)][r["seed"]].append(x)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--para", default=None, help="folder of the rewritten pages (default <results>/paraphrase)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="writers.json")
    a = ap.parse_args()
    res = Path(a.results)
    para = Path(a.para) if a.para else res / "paraphrase"
    out: dict = {"models": {}, "registered": {}, "pages": {}, "pooled_with_item28": {}}
    for f in sorted(para.glob("*_by_*.json")):
        d = json.loads(f.read_text())
        notes = [t for p in d["pages"].values() for t in p["notes"]]
        out["pages"][f.stem] = {**d.get("stats", {}), "bare_share": float(np.mean([bool(BARE.match(t)) for t in notes]))}
    arms, pooled = {}, {}
    for model, (tag, tag28, short, usual) in MODELS.items():
        recs = collect(res / f"swarm_{tag}", model, short, usual)
        if not recs:
            continue
        arms[model] = Arms(recs)
        both = collect(res / f"swarm_{tag}", model, short, usual)
        for k, v in collect(res / f"swarm_{tag28}", model, short, usual).items():
            if k in ARMS:
                for s, xs in v.items():
                    both[k][s] += xs
        pooled[model] = Arms(both)
    seeds = sorted(set().union(*[set(A.seeds) for A in arms.values()])) if arms else []
    rng = np.random.default_rng(0)
    boots = [rng.choice(seeds, size=len(seeds)) for _ in range(a.reps)]
    for model, A in arms.items():
        m = out["models"][model] = {"arms": {}, "differences": {}}
        print(f"== {model} ({MODELS[model][0]}): late holders that read the planted title in round 1")
        for k in (k for k in ARMS if k in A.recs):
            xs = [x for v in A.recs[k].values() for x in v]
            p, ci = A.rate(k), pct([A.rate(k, b) for b in boots])
            m["arms"][k] = {"games": len(xs), "treated": sum(x["treated"] for x in xs), "post2": p, "post2_ci": ci}
            print(f"   {k:9s} {len(xs):4d} games {m['arms'][k]['treated']:4d} read it  post {100 * p:5.1f} "
                  f"[{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]")
        pairs = [(hi, lo) for mod, hi, lo, _, _ in REGISTERED.values() if mod == model] + REPORTED
        for hi, lo in dict.fromkeys(pairs):
            if hi in A.recs and lo in A.recs:
                f = lambda s, hi=hi, lo=lo: A.rate(hi, s) - A.rate(lo, s)     # noqa: E731
                d, ci = f(None), pct([f(b) for b in boots])
                m["differences"][f"{hi} - {lo}"] = {"d": d, "ci": ci}
                print(f"   {hi + ' - ' + lo:20s} {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, {100 * ci[1]:+6.1f}] points")
        P = pooled[model]
        out["pooled_with_item28"][model] = {k: P.rate(k) for k in ARMS if k in P.recs}
        print("   pooled with item 28: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in out["pooled_with_item28"][model].items()))
    for lab, (model, hi, lo, sides, text) in REGISTERED.items():
        g = out["models"].get(model, {}).get("differences", {}).get(f"{hi} - {lo}")
        if g:
            met = g["ci"][0] > 0 or (sides == 2 and g["ci"][1] < 0)
            out["registered"][lab] = {"test": text, "sides": sides, **g, "met": bool(met)}
            print(f"registered ({lab}) {text}: {'met' if met else 'not met'}")
    auth = res / "writers_authorship.json"
    if auth.is_file():
        out["authorship"] = json.loads(auth.read_text())
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
