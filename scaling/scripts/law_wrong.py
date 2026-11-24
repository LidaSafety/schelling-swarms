#!/usr/bin/env python3
"""The title law with wrong guesses: collusion when copies without an answer also post guesses for it.

  .venv/bin/python scripts/law_wrong.py results --extra ../wiki_game/data/swarm4round [more folders]
      writes results/law_wrong.json and prints measured against predicted collusion per chance (CR-v2)

scripts/title_law.py gives every seeker of the base runs (the pairs) a coverage pi: the chance that one holder's note
with the value sits on the title the seeker read, in time. Copies that lack the answer also post guesses (Qwen3-4B in
about 92% of rounds, 14B in about 40%), so in big swarms a found value often sits beside wrong numbers and is used less.
Here every seeker also gets omega: the share of copies lacking that question, in other base runs, whose wrong guess
for it sits on the title the seeker read, in time (a wrong note as scripts/analyze_posting.py judges it, written in the
question's round). With h holders and g other copies lacking the answer, picking titles independently, the seeker
sees R ~ Binomial(h, pi) right notes and W ~ Binomial(g, omega) wrong ones, and

    CR = mean over seekers of  u * P(R >= 1, W = 0)  +  E[ R / (R + W) ; R >= 1, W >= 1 ]

u is the base runs' uptake (on clean pages: in pairs no found value sits beside a wrong one). Beside wrong numbers the
copy is assumed to pick a value in proportion to the notes showing it: nothing is fitted, because pairs never show a
seeker a right and a wrong note together. Also predicted: the share of found values beside a wrong number,
P(R >= 1, W >= 1) / P(R >= 1). One model per swarm, the fixed prompt, 2 answers each; base: each model's pairs.
The same with the rule left out of the prompt (--rules silent: the ruleoff tags, from pairs without the rule in
scripts/run_ruleoff_pairs.sh), and, as a cross-test, those cells from the pairs with the rule in force.
95% intervals resample the base runs (predictions) and the target runs (measurements).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_posting import judge, notes_of, run_counts  # noqa: E402
from predict_swarm import template, traces  # noqa: E402
from title_law import LATE, cells, condition, fixed, label, load, pct  # noqa: E402

MODELS = (("qwen3-4b", "Qwen3-4B"), ("qwen3-8b", "Qwen3-8B"), ("qwen3-14b", "Qwen3-14B"), ("qwen3-32b", "Qwen3-32B"),
          ("gemma-3-27b", "Gemma 3 27B"))


def wrong_titles(r: dict) -> dict[tuple[str, int], dict[str, int]]:
    """(author, question index) -> {templated title: round index} of the author's wrong guesses for that question."""
    qs = r["questions"]
    truth = {a["state"]: a["truth"] for ag in r["agents"].values() for a in ag["answers"]}
    out: dict = defaultdict(dict)
    for rk, au, t, x in notes_of(r):
        if 1 <= rk <= len(qs) and qs[rk - 1] not in r["holds"][au] and judge(t, x, qs[rk - 1], truth) == "wrong":
            out[(au, rk - 1)].setdefault(template(t, qs), rk - 1)
    return out


class WrongBase:
    """Per question: for every seeker of the base runs, which holders' right notes and which non-holders' wrong
    guesses (in other runs) sit on a title it read in time."""

    def __init__(self, rows: list[dict]):
        self.rows, self.q = rows, rows[0]["n_questions"]
        seen = used = 0
        for r in rows:
            for ag in r["agents"].values():
                for a in ag["answers"]:
                    if a["chance"] and a["seen"]:
                        seen += 1
                        used += a.get("used_chosen", a["used_other"])
        self.uptake = used / seen if seen else 0.0
        seekers, holders, guessers = [], [], []
        for ri, r in enumerate(rows):
            tr, names, wt = traces(r), list(r["events"]), wrong_titles(r)
            for ci, (holds, steps) in enumerate(tr):
                for q in range(self.q):
                    if q in holds:
                        first: dict[str, int] = {}
                        for s, (t, carried, _) in enumerate(steps):
                            if t and q in carried and t not in first:
                                first[t] = s
                        holders.append((ri, q, first))
                    else:
                        guessers.append((ri, q, wt.get((names[ci], q), {})))
                        if any(q in h for k, (h, _) in enumerate(tr) if k != ci):
                            seekers.append((ri, q, [steps[s][2] for s in range(q + 1)]))
        self.cover = {}
        for q in range(self.q):
            S = [s for s in seekers if s[1] == q]
            mats = []
            for group in ([h for h in holders if h[1] == q], [g for g in guessers if g[1] == q]):
                first = defaultdict(lambda: np.full(len(group), LATE, dtype=np.int16))
                for y, (_, _, ft) in enumerate(group):
                    for t, s in ft.items():
                        first[t][y] = s
                M = np.zeros((len(S), len(group)), dtype=bool)
                for x, (_, _, reads) in enumerate(S):
                    for rnd, t in enumerate(reads):
                        if t is not None and t in first:
                            M[x] |= first[t] <= rnd
                mats.append((np.array([g[0] for g in group]), M))
            self.cover[q] = (np.array([s[0] for s in S]), mats)

    def rates(self, w: np.ndarray | None = None) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """q -> (pi, omega, seeker weight); notes of the seeker's own run excluded."""
        w = np.ones(len(self.rows)) if w is None else w
        out = {}
        for q, (srun, mats) in self.cover.items():
            vals = []
            for grun, M in mats:
                gw = w[grun][None, :] * (srun[:, None] != grun[None, :])
                tot = gw.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    vals.append(np.where(tot > 0, (M * gw).sum(1) / np.maximum(tot, 1e-12), np.nan))
            out[q] = (vals[0], vals[1], w[srun])
        return out


def binom(n: int, p: np.ndarray) -> np.ndarray:
    """pmf of Binomial(n, p) for every p: shape (len(p), n + 1)."""
    k = np.arange(n + 1)
    c = np.array([comb(n, j) for j in k], dtype=float)
    p = p[:, None]
    return c * p ** k * (1 - p) ** (n - k)


def seeker_terms(pi, om, u, h, g):
    """Per seeker: P(found), P(found beside a wrong guess), expected use."""
    PR, PW = binom(h, pi), binom(g, om)
    R, W = np.arange(h + 1)[:, None], np.arange(g + 1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(R + W > 0, R / np.maximum(R + W, 1), 0.0)
    share[0, :] = 0.0
    found = 1 - PR[:, 0]
    beside = found * (1 - PW[:, 0])
    use = u * found * PW[:, 0] + np.einsum("sr,sw,rw->s", PR[:, 1:], PW[:, 1:], share[1:, 1:])
    return found, beside, use


def target_chances(rows: list[dict]) -> Counter:
    """(question index, holders, other copies lacking the answer) -> number of chances."""
    c = Counter()
    for r in rows:
        names, qs = list(r["events"]), r["questions"]
        holds = [set(r["holds"][n]) for n in names]
        for ci, n in enumerate(names):
            for a in r["agents"][n]["answers"]:
                if a["chance"]:
                    h = sum(a["state"] in holds[k] for k in range(len(names)) if k != ci)
                    c[(qs.index(a["state"]), h, len(names) - 1 - h)] += 1
    return c


def predict(base: WrongBase, tch: Counter, w: np.ndarray | None = None) -> dict[str, float]:
    rates = base.rates(w)
    tot, n = np.zeros(4), sum(tch.values())
    for (q, h, g), k in tch.items():
        pi, om, sw = rates[q]
        ok = ~np.isnan(pi) & ~np.isnan(om)
        f, b, use = seeker_terms(pi[ok], om[ok], base.uptake, h, g)
        ww = sw[ok] / sw[ok].sum()
        tot += k * np.array([(f * ww).sum(), (b * ww).sum(), (use * ww).sum(), (f * ww).sum() * base.uptake])
    found, beside, cr, cr_law = tot / n
    return {"found": found, "beside_wrong": beside / found if found else 0.0, "cr": cr, "cr_law": cr_law}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="law_wrong.json")
    a = ap.parse_args()
    res = Path(a.results)
    rows = load(res, [Path(e) for e in a.extra])
    out = []
    print(f"{'model':12s} {'cell':22s} {'found beside wrong':>22s} {'CR measured':>20s} {'title law':>10s} "
          f"{'with wrong guesses':>24s}")
    conds = (("forbidden", "forbidden"), ("silent", "silent"), ("silent", "forbidden"))   # (targets, base pairs)
    named = dict(MODELS)   # the open models first, by size; then any other model with pairs (API models: their id)
    others = sorted({r["model"] for r in rows if r["n_agents"] == 2 and "seat_models" not in r
                     and condition(r) in (fixed(), fixed(rule="silent"))} - set(named))
    for m, name, rule, base_rule in [(m, name, r, b) for m, name in [*MODELS, *[(o, o) for o in others]]
                                     for r, b in conds]:
        cm = cells(rows, m, fixed(rule=rule))
        base_rows = next((g for (s, n), g in cells(rows, m, fixed(rule=base_rule)).items() if n == 2), [])
        if not base_rows or not cm:
            continue
        base = WrongBase(base_rows)
        tag = {("forbidden", "forbidden"): "", ("silent", "silent"): "no rule, ",
               ("silent", "forbidden"): "no rule from rule pairs, "}[(rule, base_rule)]
        rng, prng, nb = np.random.default_rng(0), random.Random(0), len(base_rows)
        boots = [rng.multinomial(nb, [1 / nb] * nb).astype(float) for _ in range(a.reps)]
        for (src, n), g in cm.items():
            if n == 2 and rule != base_rule:
                continue
            tch = target_chances(g)
            pt, bs = predict(base, tch), [predict(base, tch, w) for w in boots]
            per = [run_counts(r) for r in g]
            obs = sum(per, Counter())
            ob = []
            for _ in range(a.reps):
                s = sum((per[prng.randrange(len(per))] for _ in per), Counter())
                ob.append(s["used"] / s["ch"])
            x = {"model": name, "rule": rule, "base_rule": base_rule, "target": tag + label(src, n), "N": n,
                 "runs": len(g), "base_runs": nb, "uptake_base": base.uptake,
                 "obs_cr": obs["used"] / obs["ch"], "obs_cr_ci": pct(ob), "obs_found": obs["seen"] / obs["ch"],
                 "obs_beside_wrong": obs["conf"] / obs["seen"] if obs["seen"] else None,
                 **{f"pred_{k}": v for k, v in pt.items()},
                 **{f"pred_{k}_ci": pct([b[k] for b in bs]) for k in ("cr", "cr_law", "beside_wrong")}}
            out.append(x)
            ok = "*" if not x["pred_cr_ci"][0] <= x["obs_cr"] <= x["pred_cr_ci"][1] else " "
            bw = "   -" if x["obs_beside_wrong"] is None else f"{100 * x['obs_beside_wrong']:5.1f}"
            print(f"{name[:28]:28s} {x['target'][:22]:22s} {bw} vs {100 * pt['beside_wrong']:5.1f} pred   "
                  f"{100 * x['obs_cr']:5.1f} [{100 * x['obs_cr_ci'][0]:4.1f},{100 * x['obs_cr_ci'][1]:5.1f}]  "
                  f"{100 * pt['cr_law']:5.1f}    {100 * pt['cr']:5.1f} [{100 * x['pred_cr_ci'][0]:4.1f},"
                  f"{100 * x['pred_cr_ci'][1]:5.1f}] {ok}", flush=True)
    (res / a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {res / a.out}\n* measured outside the interval of the law with wrong guesses")


if __name__ == "__main__":
    main()
