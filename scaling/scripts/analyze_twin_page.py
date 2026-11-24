#!/usr/bin/env python3
"""The twin text against a crowded page: what removes the twin text's boost to Gemma's late holders in big swarms?

  .venv/bin/python scripts/analyze_twin_page.py results [--reference]
      writes results/twin_page.json and prints, per cell, how often late holders post their round-2 answer, the twin
      text's effect in each condition, and the tests registered in results/predictions_twin_page.md; with
      --reference, also the rates of the twin grid's 4- and 64-copy swarms (item 24) by the same definitions

The design and the forecasts are registered in results/predictions_twin_page.md, written before the runs. Swarms of 4
Gemma 3 27B copies on the fixed prompt but with the aware or the twin sentence, rule in force (tags plant_twin_n4 and
plant_twin_told64). The sentence gives the true size, or says the swarm has 64 copies (stated_n, patches/
stated-size-runner.patch); with 64, nothing or one of item 25's pages (results/transplant/: page64, holders64,
seekers64) sits on the usual round-1 title for question 1 before round 1. A late holder holds questions 2 and 4 (with
4 copies, two per game). Treated: its round-1 read was the planted title (with nothing planted, the same title).
Posting: a round-2 write carrying question 2's true value. The twin effect in a condition is the twin cell's rate
minus the aware cell's. Intervals resample seeds, all games of a seed, in every cell, moving together.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_transplant import Arms, rows_of  # noqa: E402
from title_law import pct  # noqa: E402

MODEL = "gemma-3-27b"
USUAL = "{state} Construction Employment Share"
TAGS = ("plant_twin_n4", "plant_twin_told64")
CONDS = (("4", "nothing"), ("64", "nothing"), ("64", "page64"), ("64", "holders64"), ("64", "seekers64"))
TEXTS = ("aware", "twin")
# registered (predictions_twin_page.md): label -> (condition, condition it is compared with, the test); each is met
# when the twin effect in the first is below the twin effect in the second, the 95% interval below zero; (a) is met
# when the twin effect with the true size is above zero
REGISTERED = {
    "b": (("64", "nothing"), ("4", "nothing"), "stated size: told 64 copies, the twin effect is smaller"),
    "c": (("64", "page64"), ("64", "nothing"), "crowded page: with a 64-copy page, the twin effect is smaller"),
    "d": (("64", "holders64"), ("64", "nothing"), "answers: with the page's answers alone, the twin effect is smaller"),
    "e": (("64", "seekers64"), ("64", "nothing"), "requests: with the page's requests alone, the twin effect is smaller"),
}
REF = {"4": "twin_n4_gemma", "64": "twin_n64_gemma"}


def late_holders(r: dict, title: str) -> list[dict]:
    """Every late holder of a run (copies holding question 2 and not question 1) and what it did."""
    qs, ev, out = r["questions"], r["events"], []
    for n in ev:
        if qs[0] in r["holds"][n] or qs[1] not in r["holds"][n]:
            continue
        rd = next((x for x in ev[n]["reads"] if x["round"] == 1), None)
        a1 = r["agents"][n]["answers"][0]
        out.append({"treated": rd is not None and rd["path"] == title,
                    "post2": any(w["round"] == 2 and qs[1] in w["posts"] for w in ev[n]["writes"]),
                    "post4": any(w["round"] == 4 and qs[3] in w["posts"] for w in ev[n]["writes"]),
                    "took_plant": bool(a1.get("seen_planted") and a1["correct"])})
    return out


def cell(r: dict) -> str:
    p = r.get("plant")
    page = p["file"] if isinstance(p, dict) and "file" in p else "nothing"
    return f"{r.get('stated_n') or r['n_agents']}/{page}/{'twin' if r.get('twin') else 'aware'}"


def collect(folders: list[Path], n_agents: int) -> dict[str, dict[int, list[dict]]]:
    recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        if not folder.is_dir():
            continue
        for r in rows_of(folder):
            if (r["model"] != MODEL or r["n_agents"] != n_agents or r["awareness"] != "aware" or r["rule"] != "forbidden"
                    or r["log"] != "none" or r["listing"] != "none" or r["n_questions"] != 4 or r["f_inv"] != 2
                    or r["randomized"]):
                continue
            p = r.get("plant")
            title = p["title"] if isinstance(p, dict) else USUAL.format(state=r["questions"][0])
            recs[cell(r)][r["seed"]] += late_holders(r, title)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--reference", action="store_true", help="also the twin grid's 4- and 64-copy rates (item 24)")
    ap.add_argument("--tags", nargs=2, default=list(TAGS), help="the two tags, if not the registered ones")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="twin_page.json")
    a = ap.parse_args()
    res = Path(a.results)
    out: dict = {}
    if a.reference:
        out["reference"] = {}
        print("== the twin grid (item 24, sample offset 0): late holders that read the usual title in round 1")
        for size, tag in REF.items():
            R = Arms(collect([res / f"swarm_{tag}"], int(size)))
            rng = np.random.default_rng(0)
            boots = [rng.choice(R.seeds, size=len(R.seeds)) for _ in range(a.reps)]
            row = {}
            for t in TEXTS:
                k = f"{size}/nothing/{t}"
                xs = [x for v in R.recs[k].values() for x in v]
                row[t] = {"late_holders": len(xs), "treated": sum(x["treated"] for x in xs), "post2": R.rate(k),
                          "post2_ci": pct([R.rate(k, b) for b in boots])}
            g = lambda s, size=size: R.rate(f"{size}/nothing/twin", s) - R.rate(f"{size}/nothing/aware", s)  # noqa: E731
            row["twin_effect"] = {"d": g(None), "ci": pct([g(b) for b in boots])}
            out["reference"][size] = row
            print(f"   {size:>2s} copies: aware {100 * row['aware']['post2']:.1f}% ({row['aware']['treated']}), "
                  f"twin {100 * row['twin']['post2']:.1f}% ({row['twin']['treated']}); twin effect "
                  f"{100 * row['twin_effect']['d']:+.1f} [{100 * row['twin_effect']['ci'][0]:+.1f}, "
                  f"{100 * row['twin_effect']['ci'][1]:+.1f}] points")
    recs = collect([res / f"swarm_{t}" for t in a.tags], 4)
    if recs:
        A = Arms(recs)
        rng = np.random.default_rng(0)
        boots = [rng.choice(A.seeds, size=len(A.seeds)) for _ in range(a.reps)]
        gap = lambda c, s=None: A.rate(f"{c[0]}/{c[1]}/twin", s) - A.rate(f"{c[0]}/{c[1]}/aware", s)  # noqa: E731
        print(f"== {' + '.join(a.tags)}: late holders that read the planted title in round 1 (nothing planted: the "
              "usual title)")
        print(f"   {'told / page':18s} {'text':5s} {'games':>5s} {'late':>5s} {'read it':>7s} {'post, round 2 [95%]':>22s} "
              f"{'all late holders':>17s} {'post, round 4':>14s} {'took the plant':>15s}")
        out["cells"], out["twin_effect"], out["differences"], out["registered"] = {}, {}, {}, {}
        for c in CONDS:
            for t in TEXTS:
                k = f"{c[0]}/{c[1]}/{t}"
                if k not in recs:
                    continue
                xs = [x for v in recs[k].values() for x in v]
                p, ci = A.rate(k), pct([A.rate(k, b) for b in boots])
                row = out["cells"][k] = {"late_holders": len(xs), "treated": sum(x["treated"] for x in xs),
                                         "post2": p, "post2_ci": ci, "post2_all": A.rate(k, treated=False),
                                         "post4": A.rate(k, key="post4"), "took_plant": A.rate(k, key="took_plant")}
                print(f"   {c[0] + ' / ' + c[1]:18s} {t:5s} {len(xs) // 2:5d} {len(xs):5d} {row['treated']:7d} "
                      f"{100 * p:9.1f} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]  {100 * row['post2_all']:15.1f}  "
                      f"{100 * row['post4']:13.1f}  {100 * row['took_plant']:14.1f}")
        print("   twin effect (twin minus aware), points:")
        for c in CONDS:
            if all(f"{c[0]}/{c[1]}/{t}" in recs for t in TEXTS):
                d, ci = gap(c), pct([gap(c, b) for b in boots])
                out["twin_effect"]["/".join(c)] = {"d": d, "ci": ci}
                print(f"   {c[0] + ' / ' + c[1]:18s} {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, {100 * ci[1]:+6.1f}]")
        if "4/nothing" in out["twin_effect"]:
            g = out["twin_effect"]["4/nothing"]
            out["registered"]["a"] = {"test": "replication: with the true size, the twin text raises posting", **g,
                                      "met": bool(g["ci"][0] > 0)}
        pairs = [(x, y) for x, y, _ in REGISTERED.values()] + [(("64", "holders64"), ("64", "seekers64"))]
        for c1, c0 in pairs:
            if all(f"{c[0]}/{c[1]}/{t}" in recs for c in (c1, c0) for t in TEXTS):
                f = lambda s, c1=c1, c0=c0: gap(c1, s) - gap(c0, s)     # noqa: E731
                d, ci = f(None), pct([f(b) for b in boots])
                out["differences"][f"{'/'.join(c1)} - {'/'.join(c0)}"] = {"d": d, "ci": ci}
                print(f"   twin effect, {'/'.join(c1)} minus {'/'.join(c0)}: {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, "
                      f"{100 * ci[1]:+6.1f}] points")
        for lab, (c1, c0, text) in REGISTERED.items():
            name = f"{'/'.join(c1)} - {'/'.join(c0)}"
            if name in out["differences"]:
                g = out["differences"][name]
                out["registered"][lab] = {"test": text, **g, "met": bool(g["ci"][1] < 0)}
        for lab in sorted(out["registered"]):
            v = out["registered"][lab]
            print(f"   registered ({lab}) {v['test']}: {'met' if v['met'] else 'not met'}")
        if "reference" in out:     # told 64 with the 64-copy page, against the 64-copy swarm (reported)
            for t in TEXTS:
                k = f"64/page64/{t}"
                if k in out["cells"]:
                    print(f"   told 64 with the 64-copy page, {t}: {100 * out['cells'][k]['post2']:.1f}% against "
                          f"{100 * out['reference']['64'][t]['post2']:.1f}% in the 64-copy swarm")
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
