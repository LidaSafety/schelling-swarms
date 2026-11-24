#!/usr/bin/env python3
"""Prediction from pairs and measured collusion, split by round half.

  python byround.py <scaling dir of the scaling branch> [--reps 200]

Reuses title_law.py / law_wrong.py of the scaling branch unchanged. Rounds 1 and 3 are questions 0 and 2 (their
holders hold an answer from round 1: early holders); rounds 2 and 4 are questions 1 and 3 (their holders searched in
round 1: late holders). The prediction of a half is law_wrong.predict over that half's chances only. Intervals (95%):
the prediction resamples the pairs, the measurement resamples the swarm's games, and the miss is their difference.
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scaling")
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()
sc = Path(a.scaling).resolve()
sys.path.insert(0, str(sc / "scripts")); sys.path.insert(0, str(sc.parent))
from law_wrong import MODELS, WrongBase, predict, target_chances  # noqa: E402
from title_law import cells, fixed, load, pct  # noqa: E402

rows = load(sc / "results", [sc.parent / "wiki_game/data/swarm4round", sc / "aaai"])
HALVES = {"all": (0, 1, 2, 3), "r13": (0, 2), "r24": (1, 3)}

def measured(r, qs_idx):
    """(uses, chances) of one game over the given question indices."""
    u = c = 0
    qs = r["questions"]
    for ag in r["agents"].values():
        for x in ag["answers"]:
            if x["chance"] and qs.index(x["state"]) in qs_idx:
                c += 1; u += x.get("used_chosen", x["used_other"])
    return u, c

out = []
for m, name in MODELS:
    cm = cells(rows, m, fixed())
    base_rows = next((g for (s, n), g in cm.items() if n == 2), [])
    base = WrongBase(base_rows)
    nb = len(base_rows)
    rng, prng = np.random.default_rng(0), random.Random(0)
    boots = [rng.multinomial(nb, [1 / nb] * nb).astype(float) for _ in range(a.reps)]
    seen_n = set()
    for (src, n), g in cm.items():
        if n in seen_n:
            continue
        seen_n.add(n)
        tch = target_chances(g)
        draws = [[prng.randrange(len(g)) for _ in g] for _ in range(a.reps)]
        row = {"model": name, "N": n, "src": src, "games": len(g)}
        for hk, qi in HALVES.items():
            sub = Counter({k: v for k, v in tch.items() if k[0] in qi})
            p = 100 * predict(base, sub)["cr"]
            pb = np.array([100 * predict(base, sub, w)["cr"] for w in boots])
            per = [measured(r, qi) for r in g]
            o = 100 * sum(u for u, _ in per) / sum(c for _, c in per)
            ob = np.array([100 * sum(per[i][0] for i in d) / sum(per[i][1] for i in d) for d in draws])
            row[hk] = {"obs": o, "pred": p, "miss": o - p, "obs_ci": pct(list(ob)), "pred_ci": pct(list(pb)),
                       "miss_ci": pct(list(ob - pb))}
        out.append(row)
        print(f"{name:12s} N={n:<3d} {src:12s} " + "  ".join(
            f"{hk}: {row[hk]['obs']:5.1f} vs {row[hk]['pred']:5.1f} ({row[hk]['miss']:+5.1f} "
            f"[{row[hk]['miss_ci'][0]:+5.1f},{row[hk]['miss_ci'][1]:+5.1f}])" for hk in HALVES), flush=True)
json.dump(out, open(Path(__file__).with_name("byround.json"), "w"), indent=1)
