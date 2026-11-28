#!/usr/bin/env python3
"""The grid test: three readers x three writers. Does each model respond most to its own model's text?

  .venv/bin/python scripts/analyze_grid.py results
      writes results/grid.json and prints the reader x writer table, each reader's own-text advantage, and the tests
      registered in results/predictions_grid.md

Readers: Gemma 3 27B (tag plant_grid_gemma) and Qwen3-8B (plant_grid_q8) pairs on the pods, gpt-oss-120b pairs through
OpenRouter (plant_grid_gptoss). Writers: Gemma 3 27B, Qwen3-8B, gpt-oss-120b, each rewriting the answer page note by
note (scripts/make_paraphrase_pages.py). Gemma and Qwen3-8B read their own game's page; gpt-oss-120b reads Gemma's,
moved to its usual title (scripts/make_reader_pages.py). Late holders, treated and posting as in
scripts/analyze_page_content.py.

For reader r and writer w, the effect e[r][w] is the posting rate after w's page minus the rate with nothing planted.
Answers raise some readers' posting and lower others', so each reader's effects are read in its own direction: s[r] is
the sign of the mean of its three effects (point estimates). A reader's own-text advantage is
SP[r] = s[r] * (e[r][own] - mean of e[r][w] over the other two writers), and the diagonal index D is the mean of SP over
the three readers: with each writer once on the diagonal, a writer's general influence and a reader's general
responsiveness cancel out of D, and what remains is a reader-by-writer match. Intervals resample seeds, all games of a
seed, for all readers and arms together.
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

READERS = {"gemma": ("gemma-3-27b", "plant_grid_gemma", "{state} Construction Employment Share"),
           "q8": ("qwen3-8b", "plant_grid_q8", "{state} construction occupation share"),
           "gptoss": ("openai/gpt-oss-120b", "plant_grid_gptoss", "{state}")}
WRITERS = ("gemma", "q8", "gptoss")
NAMES = {"gemma": "Gemma 3 27B", "q8": "Qwen3-8B", "gptoss": "gpt-oss-120b"}


def arm(r: dict, reader: str) -> str:
    p = r.get("plant")
    f = p["file"] if isinstance(p, dict) and "file" in p else None
    return "nothing" if f is None else f.removeprefix(f"{reader}_by_")


def collect(res: Path) -> dict[str, Arms]:
    out = {}
    for reader, (model, tag, usual) in READERS.items():
        folder = res / f"swarm_{tag}"
        if not folder.is_dir():
            continue
        recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows_of(folder):
            if r["model"] != model or r["n_agents"] != 2 or condition(r) != fixed():
                continue
            p = r.get("plant")
            x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
            if x:
                recs[arm(r, reader)][r["seed"]].append(x)
        out[reader] = Arms(recs)
    return out


def effects(A: dict[str, Arms], seeds=None) -> dict[str, dict[str, float]]:
    return {r: {w: A[r].rate(w, seeds) - A[r].rate("nothing", seeds) for w in WRITERS} for r in A}


def stats(A: dict[str, Arms], sign: dict[str, int], seeds=None) -> dict[str, float]:
    e = effects(A, seeds)
    sp = {r: sign[r] * (e[r][r] - np.mean([e[r][w] for w in WRITERS if w != r])) for r in e}
    out = {f"SP_{r}": v for r, v in sp.items()}
    if len(sp) == 3:
        out["D"] = float(np.mean(list(sp.values())))
    if {"q8", "gptoss"} <= set(e):
        out["W"] = 0.5 * (sign["q8"] * (e["q8"]["gemma"] - e["q8"]["gptoss"])
                          + sign["gptoss"] * (e["gptoss"]["gemma"] - e["gptoss"]["q8"]))
    for r in e:
        out[f"mean_effect_{r}"] = float(np.mean(list(e[r].values())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="grid.json")
    a = ap.parse_args()
    res = Path(a.results)
    A = collect(res)
    out: dict = {"cells": {}, "effects": {}, "sign": {}, "stats": {}, "registered": {}}
    if not A:
        print("no grid runs found")
        return
    seeds = sorted(set().union(*[set(x.seeds) for x in A.values()]))
    rng = np.random.default_rng(0)
    boots = [rng.choice(seeds, size=len(seeds)) for _ in range(a.reps)]
    e0 = effects(A)
    sign = {r: 1 if np.mean(list(e0[r].values())) >= 0 else -1 for r in e0}
    out["sign"] = sign
    print("late holders posting their round-2 answer in round 2 (%), reader x planted page:")
    print(f"   {'reader':14s} {'nothing':>16s} " + " ".join(f"{'by ' + NAMES[w]:>20s}" for w in WRITERS))
    for r, X in A.items():
        row = {k: {"post2": X.rate(k), "ci": pct([X.rate(k, b) for b in boots]),
                   "treated": sum(x["treated"] for v in X.recs[k].values() for x in v)}
               for k in ("nothing", *WRITERS) if k in X.recs}
        out["cells"][r] = row
        print(f"   {NAMES[r]:14s} " + " ".join(f"{100 * row[k]['post2']:7.1f} [{100 * row[k]['ci'][0]:4.1f},{100 * row[k]['ci'][1]:5.1f}]"
                                           .rjust(16 if k == 'nothing' else 20) for k in ("nothing", *WRITERS) if k in row))
    point = stats(A, sign)
    bs = [stats(A, sign, b) for b in boots]
    for k, v in point.items():
        ci = pct([x[k] for x in bs])
        out["stats"][k] = {"value": v, "ci": ci}
    for r in A:
        out["effects"][r] = {w: e0[r][w] for w in WRITERS}
        m = out["stats"][f"mean_effect_{r}"]
        print(f"   {NAMES[r]}: answers move posting {'up' if sign[r] > 0 else 'down'} (mean effect {100 * m['value']:+.1f} "
              f"[{100 * m['ci'][0]:+.1f}, {100 * m['ci'][1]:+.1f}]); own-text advantage SP {100 * out['stats'][f'SP_{r}']['value']:+.1f} "
              f"[{100 * out['stats'][f'SP_{r}']['ci'][0]:+.1f}, {100 * out['stats'][f'SP_{r}']['ci'][1]:+.1f}] points")
    tests = {"a": ("D", "self-preference: the diagonal index D above zero"),
             "b": ("SP_gptoss", "gpt-oss-120b responds more to its own model's text than to the other two writers'"),
             "c": ("W", "writer effect: Gemma's text moves Qwen3-8B and gpt-oss-120b more than the third writer's")}
    for lab, (k, text) in tests.items():
        if k in out["stats"]:
            v = out["stats"][k]
            out["registered"][lab] = {"test": text, **v, "met": bool(v["ci"][0] > 0)}
            print(f"registered ({lab}) {text}: {100 * v['value']:+.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] "
                  f"points, {'met' if v['ci'][0] > 0 else 'not met'}")
    auth = res / "grid_authorship.json"
    if auth.is_file():
        out["authorship"] = json.loads(auth.read_text())
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
