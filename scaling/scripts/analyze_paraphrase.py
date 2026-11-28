#!/usr/bin/env python3
"""The own-style test: do late holders post more after other copies' answers written in their own model's style?

  .venv/bin/python scripts/analyze_paraphrase.py results
      writes results/paraphrase.json and prints, per reader and arm, how often late holders post their round-2 answer,
      and the tests registered in results/predictions_paraphrase.md

Pairs of Gemma 3 27B (tag plant_style_gemma) and of Qwen3-8B (plant_style_q8) on the fixed prompt. Before round 1,
nothing or the answer page of the model's own 64-copy game sits on the pair's usual round-1 title for question 1: as
written (own), rewritten note by note by the same model (self) or by the other model (other), with the values, the
number of notes and their lengths kept (scripts/make_paraphrase_pages.py). Late holders, treated and posting as in
scripts/analyze_page_content.py. Seeds listed as failed in either rewritten page of a reader are left out of all four
of its arms. Intervals resample seeds, all games of a seed, in every arm of both readers, moving together.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_page_content import late_record  # noqa: E402
from analyze_transplant import Arms, rows_of  # noqa: E402
from title_law import condition, fixed, pct  # noqa: E402

MODELS = {"gemma-3-27b": ("plant_style_gemma", "gemma", "q8", "{state} Construction Employment Share"),
          "qwen3-8b": ("plant_style_q8", "q8", "gemma", "{state} construction occupation share")}
ARMS = ("nothing", "own", "self", "other")
REGISTERED = {"a": ("gemma-3-27b", 1, "Gemma: answers rewritten by Gemma raise posting more than the same answers "
                                      "rewritten by Qwen3-8B"),
              "b": ("qwen3-8b", 2, "Qwen3-8B: answers rewritten by Qwen3-8B and by Gemma differ")}
REPORTED = [("own", "nothing"), ("self", "nothing"), ("other", "nothing"), ("own", "self"), ("own", "other")]


def arm(r: dict, short: str, other: str) -> str:
    p = r.get("plant")
    f = p["file"] if isinstance(p, dict) and "file" in p else None
    return {None: "nothing", f"{short}_answers_q1": "own", f"{short}_by_{short}": "self",
            f"{short}_by_{other}": "other"}.get(f, f or "nothing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--para", default=None, help="folder of the rewritten pages (default <results>/paraphrase)")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="paraphrase.json")
    a = ap.parse_args()
    res = Path(a.results)
    para = Path(a.para) if a.para else res / "paraphrase"
    arms: dict[str, Arms] = {}
    out: dict = {"models": {}, "registered": {}, "pages": {}}
    for model, (tag, short, other, usual) in MODELS.items():
        folder = res / f"swarm_{tag}"
        if not folder.is_dir():
            continue
        failed = set()
        for w in (short, other):
            d = json.loads((para / f"{short}_by_{w}.json").read_text())
            failed |= {int(s) for s in d.get("failed", [])}
            out["pages"][f"{short}_by_{w}"] = d.get("stats", {})
        recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows_of(folder):
            if r["model"] != model or r["n_agents"] != 2 or condition(r) != fixed() or r["seed"] in failed:
                continue
            p = r.get("plant")
            x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
            if x:
                recs[arm(r, short, other)][r["seed"]].append(x)
        arms[model] = Arms(recs)
        out["models"][model] = {"failed_seeds": sorted(failed), "arms": {}, "differences": {}}
    seeds = sorted(set().union(*[set(A.seeds) for A in arms.values()])) if arms else []
    rng = np.random.default_rng(0)
    boots = [rng.choice(seeds, size=len(seeds)) for _ in range(a.reps)]
    for model, A in arms.items():
        m = out["models"][model]
        print(f"== {model} ({MODELS[model][0]}; {len(m['failed_seeds'])} seeds left out): late holders that read the "
              "planted title in round 1")
        for k in (k for k in ARMS if k in A.recs):
            xs = [x for v in A.recs[k].values() for x in v]
            p, ci = A.rate(k), pct([A.rate(k, b) for b in boots])
            posted = [x["bare2"] for x in xs if x["treated"] and x["post2"]]
            m["arms"][k] = {"games": len(xs), "treated": sum(x["treated"] for x in xs), "post2": p, "post2_ci": ci,
                            "post2_all": A.rate(k, treated=False), "post4": A.rate(k, key="post4"),
                            "bare_share_of_posts": float(np.mean(posted)) if posted else float("nan")}
            print(f"   {k:8s} {len(xs):4d} games {m['arms'][k]['treated']:4d} read it  post {100 * p:5.1f} "
                  f"[{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]  bare posts {100 * m['arms'][k]['bare_share_of_posts']:5.1f}")
        for hi, lo in [("self", "other")] + REPORTED:
            if hi in A.recs and lo in A.recs:
                f = lambda s, hi=hi, lo=lo: A.rate(hi, s) - A.rate(lo, s)     # noqa: E731
                d, ci = f(None), pct([f(b) for b in boots])
                m["differences"][f"{hi} - {lo}"] = {"d": d, "ci": ci}
                print(f"   {hi + ' - ' + lo:16s} {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, {100 * ci[1]:+6.1f}] points")
    for lab, (model, sides, text) in REGISTERED.items():
        g = out["models"].get(model, {}).get("differences", {}).get("self - other")
        if g:
            met = g["ci"][0] > 0 or (sides == 2 and g["ci"][1] < 0)
            out["registered"][lab] = {"test": text, "sides": sides, **g, "met": bool(met)}
            print(f"registered ({lab}) {text}: {'met' if met else 'not met'}")
    auth = res / "paraphrase_authorship.json"
    if auth.is_file():
        out["authorship"] = json.loads(auth.read_text())
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
