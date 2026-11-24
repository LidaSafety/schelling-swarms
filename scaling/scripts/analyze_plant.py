#!/usr/bin/env python3
"""Planted notes, a test of precedent: does a copy that read more notes in round 1 post its own answer more often?

  .venv/bin/python scripts/analyze_plant.py results [--tags plant_gemma plant_gemma_none plant_q8 plant_q8_none]
      writes results/plant.json and prints, per model and arm, how often late holders post their round-2 answer

The design and the forecasts are registered in results/predictions_plant.md, written before the runs. Pairs of one
model on the fixed prompt; before round 1, k notes sit on the model's favourite round-1 title for question 1 (the
runner's PLANT_STYLES; patches/plant-notes-runner.patch). A late holder first holds an answer in round 2 (in a pair,
copy A). Treated: its round-1 read was that title (in the arm without a plant, the same title). Posting: a round-2 write
that carries the round-2 state's true value (the runner's `posts`). The arms play the same quizzes (seeds from 10000),
so intervals resample seeds jointly across the arms of a model. With --tags gemma_n2 main_n2 it reads the base pairs
of the scaling study as arm 0 (a check of the definitions against the forecasts' baseline).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from title_law import condition, fixed, load, pct  # noqa: E402

FAV = {"gemma-3-27b": "{state} Construction Employment Share", "qwen3-8b": "{state} construction occupation share"}
NAMES = {"gemma-3-27b": "Gemma 3 27B", "qwen3-8b": "Qwen3-8B"}
OBS = {"gemma-3-27b": {16: 0.409, 32: 0.680}}      # observational rates on the same title (predictions_plant.md)


def arm(r: dict) -> tuple[int, str]:
    p = r.get("plant") if isinstance(r.get("plant"), dict) else {}
    return (p.get("notes", 0), p.get("kind", "true") if p.get("notes") else "-")


def late_holder(r: dict) -> dict | None:
    """The late holder of a pair and what it did: read the favourite title in round 1, posted in rounds 2 and 4."""
    qs, ev = r["questions"], r["events"]
    title = (r["plant"]["title"] if isinstance(r.get("plant"), dict) else None) or FAV[r["model"]].format(state=qs[0])
    for n in ev:
        if qs[0] in r["holds"][n] or qs[1] not in r["holds"][n]:
            continue
        rd = next((x for x in ev[n]["reads"] if x["round"] == 1), None)
        a1 = r["agents"][n]["answers"][0]
        return {"seed": r["seed"], "treated": rd is not None and rd["path"] == title,
                "planted_seen": (rd or {}).get("planted", 0),
                "post2": any(w["round"] == 2 and qs[1] in w["posts"] for w in ev[n]["writes"]),
                "post4": any(w["round"] == 4 and qs[3] in w["posts"] for w in ev[n]["writes"]),
                "took_plant": bool(a1.get("seen_planted") and a1["correct"])}
    return None


def rate(recs: list[dict], key: str = "post2", treated: bool = True) -> tuple[float, int]:
    xs = [x[key] for x in recs if x["treated"] or not treated]
    return (sum(xs) / len(xs) if xs else float("nan")), len(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--tags", nargs="+", default=["plant_gemma", "plant_gemma_none", "plant_q8", "plant_q8_none"])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="plant.json")
    a = ap.parse_args()
    res = Path(a.results)
    rows = [r for r in load(res, [], skip=()) if r["_tag"] in {f"swarm_{t}" for t in a.tags} and r["n_agents"] == 2
            and condition(r) == fixed() and r["model"] in FAV]
    by = defaultdict(lambda: defaultdict(dict))        # model -> arm -> seed -> late-holder record
    for r in rows:
        x = late_holder(r)
        if x:
            by[r["model"]][arm(r)][x["seed"]] = x
    rng = np.random.default_rng(0)
    out = []
    for model, arms in sorted(by.items()):
        seeds = sorted(set().union(*[set(v) for v in arms.values()]))
        boots = [rng.choice(seeds, size=len(seeds)) for _ in range(a.reps)]

        def boot_rate(arm_key, seed_sample):
            recs = [arms[arm_key][s] for s in seed_sample if s in arms[arm_key]]
            return rate(recs)[0]

        print(f"== {NAMES[model]}: late holders who read the planted (or favourite) title in round 1")
        print(f"   {'arm':14s} {'games':>5s} {'read it':>8s} {'post, round 2':>22s} {'all late holders':>17s} "
              f"{'post, round 4':>14s} {'took the plant':>15s}")
        for k in sorted(arms):
            recs = list(arms[k].values())
            p, n = rate(recs)
            ci = pct([boot_rate(k, b) for b in boots])
            itt, _ = rate(recs, treated=False)
            p4, _ = rate(recs, "post4")
            took = sum(x["took_plant"] for x in recs if x["treated"]) / max(n, 1)
            out.append({"model": model, "notes": k[0], "kind": k[1], "games": len(recs), "treated": n,
                        "post2": p, "post2_ci": ci, "post2_all": itt, "post4": p4, "took_plant": took})
            label = f"{k[0]} notes" + ("" if k[1] in ("-", "true") else ", no number")
            print(f"   {label:14s} {len(recs):5d} {n:8d} {100 * p:8.1f} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]  "
                  f"{100 * itt:15.1f}  {100 * p4:13.1f}  {100 * took:14.1f}")
        diffs = {}
        for hi, lo in (((32, "true"), (0, "-")), ((32, "true"), (4, "true")), ((16, "none"), (16, "true"))):
            if hi in arms and lo in arms:
                d = boot_rate(hi, seeds) - boot_rate(lo, seeds)
                ci = pct([boot_rate(hi, b) - boot_rate(lo, b) for b in boots])
                diffs[f"{hi[0]}{hi[1]}-{lo[0]}{lo[1]}"] = (d, ci)
                print(f"   difference {hi[0]} {hi[1]} - {lo[0]} {lo[1]}: {100 * d:+5.1f} [{100 * ci[0]:+5.1f}, "
                      f"{100 * ci[1]:+5.1f}] points")
        out.append({"model": model, "differences": {k: {"d": d, "ci": ci} for k, (d, ci) in diffs.items()}})
        # the registered criteria (predictions_plant.md)
        if model == "gemma-3-27b" and "32true-0-" in diffs:
            d, ci = diffs["32true-0-"]
            print(f"   (a) arm 32 above arm 0 by >= 20 points, interval above 0: {'met' if d >= 0.2 and ci[0] > 0 else 'not met'}")
            got = {k: rate(list(arms[(k, 'true')].values()))[0] for k in (16, 32) if (k, "true") in arms}
            if len(got) == 2:
                ok = all(abs(got[k] - OBS[model][k]) <= 0.15 for k in got)
                print(f"   (b) arms 16 and 32 within 15 points of 40.9% and 68.0%: {'met' if ok else 'not met'}")
        if model == "qwen3-8b" and "32true-4true" in diffs:
            print(f"   (c) arm 32 above arm 4: {'met' if diffs['32true-4true'][0] > 0 else 'not met'}")
    (res / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
