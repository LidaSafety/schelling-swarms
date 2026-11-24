#!/usr/bin/env python3
"""Transplanted pages: which part of a crowded round-1 page makes a late holder post its own answer?

  .venv/bin/python scripts/analyze_transplant.py results [--source aaai]
      writes results/transplant.json and prints, per arm, how often late holders post their round-2 answer, and the
      tests registered in results/predictions_transplant.md; with --source, also the swarm rates they compare with

The design and the forecasts are registered in results/predictions_transplant.md, written before the runs. Pairs of
Gemma 3 27B on the fixed prompt (tag plant_transplant). Before round 1, nothing (arm 0), a recorded round-1 page of a
64-copy game (page64), its holders' notes only (holders64), its seekers' notes only (seekers64) or the page of a
32-copy game (page32) sits on the pair's usual round-1 title for question 1 (results/transplant/*.json, written by
scripts/make_transplant_pages.py; patches/plant-pages-runner.patch). Every seed is played three times (sample offsets
0, 1 and 2). A late holder first holds an answer in round 2 (in a pair, copy A). Treated: its round-1 read was the
planted title (in arm 0, the same title). Posting: a round-2 write that carries the round-2 state's true value (the
runner's `posts`). Intervals resample seeds, all games of a seed, in every arm, moving together.

With --source (a folder holding the aaai branch's swarm_gemma_ladder_n32 and swarm_gemma_ladder_n64, as for
make_transplant_pages.py), it also gives the rates the planted pages are compared with: in those games, the late
holders that read the main round-1 page (the title most copies wrote under in round 1), each game counted once as each
page is in the pairs, and pooled over late holders; and, on the usual title, the base pairs (gemma_n2) and item 19's
arm without a plant (plant_gemma).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_swarm import normalize  # noqa: E402
from title_law import condition, fixed, pct  # noqa: E402

MODEL = "gemma-3-27b"
USUAL = "{state} Construction Employment Share"
ARMS = ("0", "page64", "holders64", "seekers64", "page32")
REF = {"page64": 0.673, "page32": 0.400}     # the swarm rates on the same pages, as registered
MARGIN = 0.10                                # (a): within 10 points


def rows_of(folder: Path) -> list[dict]:
    """The runs of one result folder (swarm_<tag>), failed runs left out."""
    out = []
    for f in sorted((folder / "raw").glob("*.jsonl*")):
        for line in (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)):
            r = json.loads(line)
            if "error" not in r:
                out.append(normalize(r))
    return out


def arm(r: dict) -> str:
    p = r.get("plant")
    return p["file"] if isinstance(p, dict) and "file" in p else "0"


def late_holder(r: dict, title: str) -> dict | None:
    """The late holder of a pair and what it did: read `title` in round 1, posted in rounds 2 and 4."""
    qs, ev = r["questions"], r["events"]
    for n in ev:
        if qs[0] in r["holds"][n] or qs[1] not in r["holds"][n]:
            continue
        rd = next((x for x in ev[n]["reads"] if x["round"] == 1), None)
        a1 = r["agents"][n]["answers"][0]
        return {"seed": r["seed"], "treated": rd is not None and rd["path"] == title,
                "planted": (rd or {}).get("planted", 0),
                "post2": any(w["round"] == 2 and qs[1] in w["posts"] for w in ev[n]["writes"]),
                "post4": any(w["round"] == 4 and qs[3] in w["posts"] for w in ev[n]["writes"]),
                "took_plant": bool(a1.get("seen_planted") and a1["correct"])}
    return None


class Arms:
    """Late-holder records by arm and seed, with rates on any resample of the seeds."""

    def __init__(self, recs: dict[str, dict[int, list[dict]]]):
        self.recs = recs
        self.seeds = sorted(set().union(*[set(v) for v in recs.values()]))

    def rate(self, a: str, seeds=None, key: str = "post2", treated: bool = True) -> float:
        xs = [x[key] for s in (self.seeds if seeds is None else seeds) for x in self.recs[a].get(s, [])
              if x["treated"] or not treated]
        return sum(xs) / len(xs) if xs else float("nan")


def reference(source: Path, res: Path, rng: np.random.Generator, reps: int) -> dict:
    """The swarm rates (main round-1 page) and the base pairs' (usual title), each with a 95% interval."""
    out = {}
    for n in (64, 32):
        per = []                                  # per game: (posted, late holders that read the main page)
        for r in rows_of(source / f"swarm_gemma_ladder_n{n}"):
            ev, qs = r["events"], r["questions"]
            top = Counter(w["path"] for a in ev for w in ev[a]["writes"]
                          if w["round"] == 1 and w.get("path")).most_common(1)[0][0]
            y = [any(w["round"] == 2 and qs[1] in w["posts"] for w in ev[a]["writes"]) for a in ev
                 if qs[0] not in r["holds"][a] and qs[1] in r["holds"][a]
                 and any(x["round"] == 1 and x["path"] == top for x in ev[a]["reads"])]
            if y:
                per.append((sum(y), len(y)))
        P = np.array(per, float)
        game = lambda Q: float(np.mean(Q[:, 0] / Q[:, 1]))          # noqa: E731
        pool = lambda Q: float(Q[:, 0].sum() / Q[:, 1].sum())       # noqa: E731
        boots = [P[rng.integers(0, len(P), len(P))] for _ in range(reps)]
        out[f"page{n}"] = {"games": len(P), "late_holders": int(P[:, 1].sum()), "by_game": game(P),
                           "by_game_ci": pct([game(Q) for Q in boots]), "pooled": pool(P),
                           "pooled_ci": pct([pool(Q) for Q in boots])}
    for tag, label in (("gemma_n2", "pairs"), ("plant_gemma", "plant_gemma arm 0")):
        xs = [late_holder(r, USUAL.format(state=r["questions"][0])) for r in rows_of(res / f"swarm_{tag}")
              if r["model"] == MODEL and r["n_agents"] == 2 and condition(r) == fixed()
              and not (isinstance(r.get("plant"), dict) and r["plant"].get("notes"))]
        ys = np.array([x["post2"] for x in xs if x and x["treated"]], float)
        boots = [ys[rng.integers(0, len(ys), len(ys))].mean() for _ in range(reps)]
        out[label] = {"late_holders": len(ys), "posted": int(ys.sum()), "rate": float(ys.mean()), "ci": pct(boots)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--tag", default="plant_transplant")
    ap.add_argument("--source", help="folder holding the aaai branch's swarm_gemma_ladder_n32 and _n64")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="transplant.json")
    a = ap.parse_args()
    res = Path(a.results)
    rng = np.random.default_rng(0)
    out: dict = {}
    if a.source:
        ref = out["reference"] = reference(Path(a.source), res, rng, a.reps)
        print("== the rates the planted pages compare with: late holders posting their round-2 answer in round 2")
        for n in (64, 32):
            x = ref[f"page{n}"]
            print(f"   {n} copies, the main round-1 page ({x['games']} games, {x['late_holders']} late holders): "
                  f"{100 * x['by_game']:.1f}% [{100 * x['by_game_ci'][0]:.1f}, {100 * x['by_game_ci'][1]:.1f}] "
                  f"each game counted once, {100 * x['pooled']:.1f}% [{100 * x['pooled_ci'][0]:.1f}, "
                  f"{100 * x['pooled_ci'][1]:.1f}] pooled")
        for label in ("pairs", "plant_gemma arm 0"):
            x = ref[label]
            print(f"   {label}, the usual title: {x['posted']} of {x['late_holders']} = {100 * x['rate']:.1f}% "
                  f"[{100 * x['ci'][0]:.1f}, {100 * x['ci'][1]:.1f}]")
    folder = res / f"swarm_{a.tag}"
    if folder.is_dir():
        recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows_of(folder):
            if r["model"] != MODEL or r["n_agents"] != 2 or condition(r) != fixed():
                continue
            p = r.get("plant")
            x = late_holder(r, p["title"] if isinstance(p, dict) else USUAL.format(state=r["questions"][0]))
            if x:
                recs[arm(r)][r["seed"]].append(x | {"offset": r.get("sample_offset", 0)})
        A = Arms(recs)
        boots = [rng.choice(A.seeds, size=len(A.seeds)) for _ in range(a.reps)]
        have = [k for k in ARMS if k in recs]
        print(f"== {a.tag}: late holders that read the planted title in round 1 (arm 0: the usual title)")
        print(f"   {'arm':10s} {'games':>5s} {'read it':>7s} {'post, round 2 [95%]':>22s} {'all late holders':>17s} "
              f"{'post, round 4':>14s} {'took the plant':>15s}")
        out["arms"] = {}
        for k in have:
            xs = [x for v in recs[k].values() for x in v]
            n = sum(x["treated"] for x in xs)
            p, ci = A.rate(k), pct([A.rate(k, b) for b in boots])
            row = {"games": len(xs), "treated": n, "post2": p, "post2_ci": ci,
                   "post2_all": A.rate(k, treated=False), "post4": A.rate(k, key="post4"),
                   "took_plant": A.rate(k, key="took_plant")}
            if k != "0":     # within the arm, pages with fewer or more planted notes than the arm's median
                med = float(np.median([x["planted"] for x in xs if x["treated"]]))
                row["post2_by_page_length"] = {
                    f"below {med:.0f}": np.mean([x["post2"] for x in xs if x["treated"] and x["planted"] < med]),
                    f"{med:.0f} or more": np.mean([x["post2"] for x in xs if x["treated"] and x["planted"] >= med])}
            out["arms"][k] = row
            print(f"   {k:10s} {len(xs):5d} {n:7d} {100 * p:9.1f} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]  "
                  f"{100 * row['post2_all']:15.1f}  {100 * row['post4']:13.1f}  {100 * row['took_plant']:14.1f}")
        contrasts = {"page64 - holders64": {"page64": 1, "holders64": -1},
                     "page64 - seekers64": {"page64": 1, "seekers64": -1},
                     "page32 - seekers64": {"page32": 1, "seekers64": -1},
                     "holders64 - 0": {"holders64": 1, "0": -1}, "seekers64 - 0": {"seekers64": 1, "0": -1},
                     "page64 - holders64 - seekers64 + 0": {"page64": 1, "holders64": -1, "seekers64": -1, "0": 1}}
        out["differences"] = {}
        for name, w in contrasts.items():
            if all(k in recs for k in w):
                f = lambda s, w=w: sum(c * A.rate(k, s) for k, c in w.items())    # noqa: E731
                d, ci = f(None), pct([f(b) for b in boots])
                out["differences"][name] = {"d": d, "ci": ci}
                print(f"   {name:36s} {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, {100 * ci[1]:+6.1f}] points")
        D, R = out["differences"], out["arms"]
        reg = {}
        if all(k in R for k in REF):
            reg["a"] = {"page64": R["page64"]["post2"], "page32": R["page32"]["post2"],
                        "met": bool(all(abs(R[k]["post2"] - REF[k]) <= MARGIN for k in REF))}
        if {"page64 - holders64", "page64 - seekers64"} <= set(D):
            reg["b"] = {"met": bool(all(D[k]["d"] > 0 and D[k]["ci"][0] > 0
                                        for k in ("page64 - holders64", "page64 - seekers64")))}
        if "page32 - seekers64" in D:
            reg["c"] = {"met": bool(D["page32 - seekers64"]["d"] > 0 and D["page32 - seekers64"]["ci"][0] > 0)}
        out["registered"] = reg
        text = {"a": "page64 within 10 points of 67.3% and page32 within 10 points of 40.0%",
                "b": "page64 above holders64 and above seekers64, both intervals above zero",
                "c": "page32 above seekers64, the interval above zero"}
        for h, v in reg.items():
            print(f"   registered ({h}) {text[h]}: {'met' if v['met'] else 'not met'}")
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
