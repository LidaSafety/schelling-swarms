#!/usr/bin/env python3
"""Holders against seekers: does a bigger swarm help a seeker beyond the copies that hold its answer? Qwen3-8B, R=1.

  .venv/bin/python scripts/holders_seekers.py results
      writes results/holders_seekers.json and prints both estimates

With R=1 (each copy holds its own random 2 of the 4 answers) the number of copies holding a question varies from run to
run, so the same number of holders turns up at different swarm sizes. Per chance (a copy facing a question it does not
hold, in a round where another copy holds it): N, holders h, seekers s = N - h, round, found (it had seen the right
value, CR-v2's "seen") and used. Two estimates of what seekers add when the holders are fixed:
  matched   N against 2N at the holder counts both sizes have: the difference in found, averaged over those counts
            (weighted by chances); 95% intervals resample the runs of each size
  logistic  logit P(found) = b0 + b1 log2 h + b2 log2 s over N = 4-32: change in log-odds per doubling of holders (b1)
            and of seekers (b2), also in points at the mean; 95% intervals resample runs
Both are split into round 1 and rounds 2-4. Round 1 is a placebo: no copy has read anything before it writes and reads
in round 1, the unaware prompt does not depend on N, and only holders know the right value, so at fixed holders the
seekers cannot change found there. Whatever an estimate gives for round 1 is its own error.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from title_law import cells, fixed, load  # noqa: E402

PAIRS = ((4, 8), (8, 16), (16, 32))


def chances(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Columns N, h, s, round, found, used (one row per chance) and the run index of each row."""
    rec, run = [], []
    for (_, n), g in cells(rows, "qwen3-8b", fixed(rnd=1)).items():
        for r in g:
            names = list(r["events"])
            holds = {x: set(r["holds"][x]) for x in names}
            for x in names:
                for a in r["agents"][x]["answers"]:
                    if a["chance"]:
                        h = sum(a["state"] in holds[o] for o in names if o != x)
                        rec.append((n, h, n - h, a["round"], a["seen"], a.get("used_chosen", a["used_other"])))
                        run.append(id(r))
    ids = {v: i for i, v in enumerate(dict.fromkeys(run))}
    return np.array(rec, dtype=float), np.array([ids[v] for v in run])


def matched(R: np.ndarray, runs: np.ndarray, mask: np.ndarray, small: int, big: int, rng, reps: int = 400) -> dict:
    ms, mb = mask & (R[:, 0] == small), mask & (R[:, 0] == big)
    hs = sorted(set(R[ms, 1]) & set(R[mb, 1]))

    def diff(cnt: np.ndarray) -> float:
        num = den = 0.0
        for h in hs:
            a, b = ms & (R[:, 1] == h), mb & (R[:, 1] == h)
            wa, wb = cnt[a], cnt[b]
            if wa.sum() and wb.sum():
                w = wa.sum() + wb.sum()
                num += w * ((R[b, 4] * wb).sum() / wb.sum() - (R[a, 4] * wa).sum() / wa.sum())
                den += w
        return num / den if den else float("nan")

    point = diff(np.ones(len(R)))
    rows_of = defaultdict(list)
    for i, k in enumerate(runs):
        rows_of[k].append(i)
    ids_s, ids_b = sorted(set(runs[ms])), sorted(set(runs[mb]))
    vals = []
    for _ in range(reps):
        cnt = np.zeros(len(R))
        for ids in (ids_s, ids_b):
            for k in rng.choice(len(ids), len(ids)):
                cnt[rows_of[ids[k]]] += 1
        vals.append(diff(cnt))
    vals = np.sort(vals)
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    return {"small": small, "big": big, "holders": [int(hs[0]), int(hs[-1])], "diff": point,
            "ci": [float(vals[lo]), float(vals[hi])],
            "chances": [int((ms & np.isin(R[:, 1], hs)).sum()), int((mb & np.isin(R[:, 1], hs)).sum())]}


def logit_fit(X: np.ndarray, y: np.ndarray, it: int = 50) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(len(b))
        b += np.linalg.solve(H, X.T @ (y - p))
    return b


def logistic(R: np.ndarray, runs: np.ndarray, mask: np.ndarray, reps: int = 300, seed: int = 1) -> dict:
    sel = np.where(mask & (R[:, 0] >= 4))[0]
    design = lambda ii: np.column_stack([np.ones(len(ii)), np.log2(R[ii, 1]), np.log2(R[ii, 2])])
    b = logit_fit(design(sel), R[sel, 4])
    ids = sorted(set(runs[sel]))
    rows_of = defaultdict(list)
    for i in sel:
        rows_of[runs[i]].append(i)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        ii = [i for k in rng.choice(len(ids), len(ids)) for i in rows_of[ids[k]]]
        bs.append(logit_fit(design(ii), R[ii, 4]))
    bs = np.sort(np.array(bs), axis=0)
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    p = R[sel, 4].mean()
    slope = lambda j: {"log_odds": float(b[j]), "ci": [float(bs[lo, j]), float(bs[hi, j])],
                       "points_at_mean": float(b[j] * p * (1 - p)),
                       "points_ci": [float(bs[lo, j] * p * (1 - p)), float(bs[hi, j] * p * (1 - p))]}
    return {"chances": len(sel), "found": float(p), "holders": slope(1), "seekers": slope(2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default="holders_seekers.json")
    a = ap.parse_args()
    res = Path(a.results)
    R, runs = chances(load(res, [Path(e) for e in a.extra]))
    print(f"Qwen3-8B, R=1: {len(R)} chances in {len(set(runs))} runs")
    rng = np.random.default_rng(0)
    out = {"chances": len(R), "runs": len(set(runs)), "groups": []}
    for lab, mask in (("round 1", R[:, 3] == 1), ("rounds 2-4", R[:, 3] > 1)):
        g = {"rounds": lab, "matched": [matched(R, runs, mask, s, b, rng) for s, b in PAIRS],
             "logistic": logistic(R, runs, mask)}
        out["groups"].append(g)
        print(f"\n{lab}: found, bigger swarm minus smaller at the same holders (points)")
        for m in g["matched"]:
            print(f"   N={m['small']:<2d} vs N={m['big']:<2d}, holders {m['holders'][0]}-{m['holders'][1]}: "
                  f"{100 * m['diff']:+5.1f} [{100 * m['ci'][0]:+5.1f}, {100 * m['ci'][1]:+5.1f}]   "
                  f"chances {m['chances'][0]} vs {m['chances'][1]}")
        lg = g["logistic"]
        for k in ("holders", "seekers"):
            x = lg[k]
            print(f"   per doubling of {k:8s}: {x['log_odds']:+.2f} log-odds [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] "
                  f"= {100 * x['points_at_mean']:+.1f} points at the mean [{100 * x['points_ci'][0]:+.1f}, "
                  f"{100 * x['points_ci'][1]:+.1f}]")
    (res / a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {res / a.out}")


if __name__ == "__main__":
    main()
