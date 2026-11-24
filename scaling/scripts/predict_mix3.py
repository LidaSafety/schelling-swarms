#!/usr/bin/env python3
"""Predictions for swarms that mix Qwen3-4B, Qwen3-8B and Gemma 3 27B, from each model's pairs.

The paper's prediction from pairs with wrong guesses (scripts/law_wrong.py), extended to several models as
scripts/title_law.py does for coverage: a seeker of model a sees R = sum_m Bin(h_m, pi_am) true values and
W = sum_m Bin(g_m, omega_am) wrong guesses, where pi_am (omega_am) is the share of model m's holders (non-holders) in
other base pairs whose true note (wrong guess) sits on a title the seeker read, in time. Then, per seeker,

    use = u_a * P(R >= 1, W = 0) + E[R / (R + W); R >= 1, W >= 1]

with u_a the uptake of model a's pairs; CR is the mean over the swarm's chances. Also the title match f of the paper's
Appendix (Gram matrix M of the paper's table), pooled and within one game.

Checks printed first: pure swarms (must match law_wrong.py) and the measured two-model mixes (4, 8 and 16 copies).

  .venv/bin/python scripts/predict_mix3.py [--aaai <folder with the aaai branch's swarm_mix16>]
      writes results/predictions_mix3.json; registered in results/predictions_mix3.md before the runs
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posting import run_counts  # noqa: E402
from law_wrong import binom, wrong_titles  # noqa: E402
from predict_swarm import traces  # noqa: E402
from title_law import LATE, Base, cells, condition, fixed, load, pct, seat_models  # noqa: E402
from title_law import predict as tl_predict  # noqa: E402

MODELS = ("qwen3-4b", "qwen3-8b", "gemma-3-27b")
SHORT = {"qwen3-4b": "4B", "qwen3-8b": "8B", "gemma-3-27b": "Gemma"}
# Title match M of the paper (Table "Title match M of the five models"): round-1 writes, pure swarms of N <= 8
M = {("qwen3-4b", "qwen3-4b"): 0.335, ("qwen3-4b", "qwen3-8b"): 0.027, ("qwen3-4b", "gemma-3-27b"): 0.047,
     ("qwen3-8b", "qwen3-8b"): 0.303, ("qwen3-8b", "gemma-3-27b"): 0.054, ("gemma-3-27b", "gemma-3-27b"): 0.640}
PLAN = [((8, 8, 0), 200), ((8, 0, 8), 100), ((6, 8, 2), 200), ((4, 4, 8), 100)]   # (4B, 8B, Gemma), games


def mm(a: str, b: str) -> float:
    return M.get((a, b), M.get((b, a)))


def title_match(counts: dict[str, int]) -> tuple[float, float, float]:
    """Pooled title match f, weighted average of the pure swarms w, and f_N for two distinct copies of one game."""
    n = sum(counts.values())
    x = {m: c / n for m, c in counts.items() if c}
    f = sum(x[a] * x[b] * mm(a, b) for a in x for b in x)
    w = sum(x[a] * mm(a, a) for a in x)
    return f, w, w - n / (n - 1) * (w - f)


class MixBase:
    """Seekers of every model in the base pairs, and which holders' true notes and which non-holders' wrong guesses of
    each model (in other pairs) sit on a title the seeker read in time."""

    def __init__(self, rows: list[dict]):
        self.rows, self.q = rows, rows[0]["n_questions"]
        seen, used = Counter(), Counter()
        for r in rows:
            ms = dict(zip(r["events"], seat_models(r)))
            for n, ag in r["agents"].items():
                for a in ag["answers"]:
                    if a["chance"] and a["seen"]:
                        seen[ms[n]] += 1
                        used[ms[n]] += a.get("used_chosen", a["used_other"])
        self.uptake = {m: used[m] / seen[m] for m in seen}
        seekers, holders, guessers = [], [], []
        for ri, r in enumerate(rows):
            tr, names, wt, ms = traces(r), list(r["events"]), wrong_titles(r), seat_models(r)
            for ci, (holds, steps) in enumerate(tr):
                for q in range(self.q):
                    if q in holds:
                        first: dict[str, int] = {}
                        for s, (t, carried, _) in enumerate(steps):
                            if t and q in carried and t not in first:
                                first[t] = s
                        holders.append((ri, ms[ci], q, first))
                    else:
                        guessers.append((ri, ms[ci], q, wt.get((names[ci], q), {})))
                        if any(q in h for k, (h, _) in enumerate(tr) if k != ci):
                            seekers.append((ri, ms[ci], q, [steps[s][2] for s in range(q + 1)]))
        models = sorted({h[1] for h in holders})
        self.cover = {}
        for q in range(self.q):
            for a in sorted({s[1] for s in seekers if s[2] == q}):
                S = [s for s in seekers if s[2] == q and s[1] == a]
                per_m = {}
                for m in models:
                    mats = []
                    for group in ([h for h in holders if h[2] == q and h[1] == m],
                                  [g for g in guessers if g[2] == q and g[1] == m]):
                        first = defaultdict(lambda: np.full(len(group), LATE, dtype=np.int16))
                        for y, (_, _, _, ft) in enumerate(group):
                            for t, s in ft.items():
                                first[t][y] = s
                        Mx = np.zeros((len(S), len(group)), dtype=bool)
                        for x, (_, _, _, reads) in enumerate(S):
                            for rnd, t in enumerate(reads):
                                if t is not None and t in first:
                                    Mx[x] |= first[t] <= rnd
                        mats.append((np.array([g[0] for g in group]), Mx))
                    per_m[m] = mats
                self.cover[(q, a)] = (np.array([s[0] for s in S]), per_m)

    def rates(self, w: np.ndarray | None = None) -> dict:
        """(q, seeker model) -> ({holder model: (pi, omega)}, seeker weight); notes of the seeker's own run excluded."""
        w = np.ones(len(self.rows)) if w is None else w
        out = {}
        for key, (srun, per_m) in self.cover.items():
            d = {}
            for m, mats in per_m.items():
                vals = []
                for grun, Mx in mats:
                    gw = w[grun][None, :] * (srun[:, None] != grun[None, :])
                    tot = gw.sum(1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        vals.append(np.where(tot > 0, (Mx * gw).sum(1) / np.maximum(tot, 1e-12), np.nan))
                d[m] = (vals[0], vals[1])
            out[key] = (d, w[srun])
        return out


def conv(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise convolution of two sets of pmfs: (S, a) * (S, b) -> (S, a + b - 1)."""
    out = np.zeros((A.shape[0], A.shape[1] + B.shape[1] - 1))
    for j in range(B.shape[1]):
        out[:, j:j + A.shape[1]] += A * B[:, [j]]
    return out


def pmf_sum(counts: dict[str, int], probs: dict[str, np.ndarray], size: int) -> np.ndarray:
    """pmf of sum_m Binomial(counts[m], probs[m]) per seeker."""
    P = np.ones((size, 1))
    for m, n in counts.items():
        if n > 0:
            P = conv(P, binom(n, np.nan_to_num(probs[m])))
    return P


def seeker_terms(PR: np.ndarray, PW: np.ndarray, u: float):
    h, g = PR.shape[1] - 1, PW.shape[1] - 1
    R, W = np.arange(h + 1)[:, None], np.arange(g + 1)[None, :]
    share = np.where(R + W > 0, R / np.maximum(R + W, 1), 0.0)
    share[0, :] = 0.0
    found = 1 - PR[:, 0]
    beside = found * (1 - PW[:, 0])
    use = u * found * PW[:, 0] + np.einsum("sr,sw,rw->s", PR[:, 1:], PW[:, 1:], share[1:, 1:])
    return found, beside, use


def predict(base: MixBase, tch: Counter, w: np.ndarray | None = None) -> dict[str, float]:
    rates = base.rates(w)
    tot, n = np.zeros(3), 0
    for (q, a, hm, gm), k in tch.items():
        d, sw = rates[(q, a)]
        ok = np.ones(len(sw), dtype=bool)
        for m in d:
            ok &= ~np.isnan(d[m][0]) & ~np.isnan(d[m][1])
        size = int(ok.sum())
        PR = pmf_sum(dict(hm), {m: d[m][0][ok] for m in d}, size)
        PW = pmf_sum(dict(gm), {m: d[m][1][ok] for m in d}, size)
        f, b, use = seeker_terms(PR, PW, base.uptake[a])
        ww = sw[ok] / sw[ok].sum()
        tot += k * np.array([(f * ww).sum(), (b * ww).sum(), (use * ww).sum()])
        n += k
    found, beside, cr = tot / n
    return {"found": found, "beside_wrong": beside / found if found else 0.0, "cr": cr}


def plan_chances(counts: dict[str, int], q: int = 4) -> Counter:
    """Chances of a planned swarm: 2 answers per copy, half of each model's copies hold questions 1 and 3, half 2 and 4
    (the runner's fixed blocks with an even count per model)."""
    tch = Counter()
    live = {m: c for m, c in counts.items() if c}
    for qi in range(q):
        for a, ca in live.items():
            hm = tuple(sorted((m, c // 2) for m, c in live.items()))
            gm = tuple(sorted((m, c // 2 - (m == a)) for m, c in live.items()))
            tch[(qi, a, hm, gm)] += ca // 2
    return tch


def run_chances(rows: list[dict]) -> Counter:
    """Chances of measured runs, with holder and non-holder counts by model."""
    tch = Counter()
    for r in rows:
        ms, names, qs = seat_models(r), list(r["events"]), r["questions"]
        holds = [set(r["holds"][n]) for n in names]
        for ci, n in enumerate(names):
            for a in r["agents"][n]["answers"]:
                if a["chance"]:
                    st = a["state"]
                    hm = Counter(ms[k] for k in range(len(names)) if k != ci and st in holds[k])
                    gm = Counter(ms[k] for k in range(len(names)) if k != ci and st not in holds[k])
                    tch[(qs.index(st), ms[ci], tuple(sorted(hm.items())), tuple(sorted(gm.items())))] += 1
    return tch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaai", help="folder holding the aaai branch's swarm_mix16 (adds its 7 mixes to the checks)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=str(SC / "results" / "predictions_mix3.json"))
    a = ap.parse_args()
    extra = [SC.parent / "wiki_game" / "data" / "swarm4round"] + ([Path(a.aaai)] if a.aaai else [])
    rows = load(SC / "results", extra)
    base_rows = []
    for m in MODELS:
        cm = cells(rows, m, fixed())
        b = cm.get(("results", 2)) or next((g for (s, n), g in cm.items() if n == 2), [])
        print(f"base pairs {SHORT[m]:6s}: {len(b)} runs from {b[0]['_tag']}")
        base_rows += b
    base = MixBase(base_rows)
    tl_base = Base(base_rows)
    print("uptake:", {SHORT[m]: round(u, 3) for m, u in base.uptake.items()})
    rng = np.random.default_rng(0)
    nb = len(base_rows)
    boots = [rng.multinomial(nb, [1 / nb] * nb).astype(float) for _ in range(a.reps)]

    def with_ci(tch):
        pt = predict(base, tch)
        bs = [predict(base, tch, w) for w in boots]
        return pt, {k: pct([b[k] for b in bs]) for k in ("cr", "beside_wrong")}

    print("\nChecks: pure swarms of 16 (law_wrong.py and the paper: 4B 37.3, 8B 56.2, Gemma 72.3)")
    for m in MODELS:
        pt, ci = with_ci(plan_chances({m: 16}))
        print(f"  {SHORT[m]:6s} N=16   predicted {100 * pt['cr']:5.1f} [{100 * ci['cr'][0]:4.1f}, {100 * ci['cr'][1]:4.1f}]")
    print("\nChecks: measured two-model mixes of Qwen3-8B and Gemma (the paper: all 11 within 2.6 points)")
    mixes = defaultdict(list)
    for r in rows:
        if "seat_models" in r and condition(r) == fixed():
            mixes[(r["n_agents"], r["model"])].append(r)
    checks = []
    for (n, lab), g in sorted(mixes.items()):
        pt, ci = with_ci(run_chances(g))
        obs = sum((run_counts(r) for r in g), Counter())
        o = obs["used"] / obs["ch"]
        checks.append({"mix": lab, "N": n, "runs": len(g), "obs_cr": o, "pred_cr": pt["cr"], "pred_cr_ci": ci["cr"]})
        flag = " " if ci["cr"][0] <= o <= ci["cr"][1] else "*"
        print(f"  {lab[:34]:34s} N={n:<3d} measured {100 * o:5.1f}   predicted {100 * pt['cr']:5.1f} "
              f"[{100 * ci['cr'][0]:4.1f}, {100 * ci['cr'][1]:4.1f}] {flag}  diff {100 * (o - pt['cr']):+5.1f}")

    print("\nPredictions for the planned swarms (16 copies, the fixed prompt, 2 answers per copy)")
    print(f"  {'mix 4B:8B:Gemma':16s} {'games':>5s}  {'title match f':>13s} {'f in one game':>13s} {'weighted avg':>12s}"
          f"  {'CR predicted [95%]':>22s}  {'found beside a guess':>20s}  {'CR, no guesses':>14s}")
    plan = []
    for (c4, c8, cg), games in PLAN:
        counts = dict(zip(MODELS, (c4, c8, cg)))
        f, w, fN = title_match(counts)
        pt, ci = with_ci(plan_chances(counts))
        tl = [(a_, q_, hm) for (q_, a_, hm, _), k in plan_chances(counts).items() for _ in range(k)]
        no_guess = tl_predict(tl_base, tl)["cr"]
        plan.append({"mix": f"qwen3-4b:{c4},qwen3-8b:{c8},gemma-3-27b:{cg}", "counts": counts, "N": 16,
                     "games": games, "title_match_pooled": f, "title_match_within_game": fN,
                     "title_match_weighted_average": w, "pred_cr": pt["cr"], "pred_cr_ci": ci["cr"],
                     "pred_found": pt["found"], "pred_beside_wrong": pt["beside_wrong"],
                     "pred_beside_wrong_ci": ci["beside_wrong"], "pred_cr_without_guesses": no_guess})
        print(f"  {c4}:{c8}:{cg:<12d} {games:5d}  {f:13.3f} {fN:13.3f} {w:12.3f}  {100 * pt['cr']:8.1f} "
              f"[{100 * ci['cr'][0]:4.1f}, {100 * ci['cr'][1]:4.1f}]  {100 * pt['beside_wrong']:12.1f} "
              f"[{100 * ci['beside_wrong'][0]:4.1f}, {100 * ci['beside_wrong'][1]:4.1f}]  {100 * no_guess:14.1f}")
    Path(a.out).write_text(json.dumps({"base_runs": {SHORT[m]: sum(r["model"] == m for r in base_rows) for m in MODELS},
                                       "uptake": base.uptake, "M": {f"{x}|{y}": v for (x, y), v in M.items()},
                                       "checks_two_model_mixes": checks, "plan": plan}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
