#!/usr/bin/env python3
"""Title law for the N-copy wiki dead drop: collusion from the spread of page titles and the number of holders.

  .venv/bin/python scripts/title_law.py results [--extra ../wiki_game/data/swarm4round] [--reps 200]
      writes results/title_law.json and prints measured against predicted collusion per chance (CR-v2)

A seeker is a copy facing a question it does not hold, in a round where another copy holds it. For every seeker of the
base runs (the family's smallest runs), coverage pi is the share of that question's holders in OTHER base runs whose
notes would have shown the seeker the value: a holder note carrying the value sits on a title the seeker read, written
no later than that read. Titles are compared after replacing each run's quiz states by their round, as
scripts/predict_swarm.py does. If copies pick titles independently, a seeker facing h holders finds the value with
chance 1 - (1 - pi)^h, so

    CR(h) = uptake * mean over seekers of [1 - (1 - pi)^h]       the title law (the whole spread of pi)
    CR(h) = uptake * [1 - (1 - m)^h],  m = mean(pi)                 the one-number law

uptake = used / seen over the base runs' chances. The two laws differ because seekers differ: a seeker that reads an
unusual title has pi near 0 and no number of holders helps it, so the title law levels off below uptake while the
one-number law climbs to it. A seeker of model a facing h_m holders of model m finds the value with
1 - prod_m (1 - pi_am)^h_m (mixed swarms). Each target is predicted chance by chance from its own holder counts, so
random splits and mixes need no special case. Not modelled: notes from non-holders, and any reaction to what a copy
reads. Intervals: predictions resample the base runs, observations resample the target runs (95%).

Families (the fixed prompt: unaware text, rule in force, no page list, no log, unless named):
  Qwen3-8B with 2 answers each, from pairs: every size here (2-128 copies), random splits, and the 1-answer swarms
    (a deliberate cross-test); 1 answer each from its 4-copy runs; 8 questions from pairs; 8 questions with 2 and 1
    answers each from their 4- and 8-copy runs, and from the 8-question pairs (cross-test); Gemma 3 27B from pairs;
    the mixed swarms from both models' pairs; the aware and twin texts from 4-copy runs.
  With --extra ../wiki_game/data/swarm4round (2-32 copies of Qwen3-4B, 14B and 32B in the same game): those models
    from their pairs, up to the 64- and 128-copy runs here. Runs in extra folders are labelled by folder name.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_swarm import SKIP_TAGS, chances, normalize, traces  # noqa: E402
from scb.io import read_jsonl  # noqa: E402

FIXED = ("unaware", 0, "none", "none", "forbidden", "none", "wiki")   # awareness, twin, log, defense, rule, listing, channel
LATE = 99      # "never posted there" (rounds are 1..8)


def load(res: Path, extra: list[Path], skip: tuple[str, ...] = SKIP_TAGS) -> list[dict]:
    """Every saved run of res and the extra folders, normalized; _src is 'results' or the extra folder's name. Folders
    starting with `skip` are left out (by default the planted-notes runs: see SKIP_TAGS)."""
    rows = []
    for d, src in [(res, "results")] + [(e, e.name) for e in extra]:
        for f in sorted(d.glob("swarm*/raw/*.jsonl*")):
            if f.parts[-3].startswith(skip):
                continue
            for r in read_jsonl(f):
                if "error" in r:
                    continue
                r = normalize(r)
                r["_src"], r["_tag"] = src, f.parts[-3]
                r["model"] = r["model"].removesuffix("-bf16")     # Qwen3-32B served at full precision
                rows.append(r)
    return rows


def condition(r: dict) -> tuple:
    """Everything but the model and N: the prompt and the game."""
    return (r["awareness"], r["twin"], r["log"], r.get("defense", "none"), r["rule"], r["listing"], r["channel"],
            r["n_questions"], r["f_inv"], r["randomized"], r.get("read_cap"), r.get("n_fillers", 2))


def fixed(q: int = 4, finv: int = 2, rnd: int = 0, aw: str = "unaware", twin: int = 0,
          rule: str = "forbidden", fillers: int = 2) -> tuple:
    return (aw, twin) + FIXED[2:4] + (rule,) + FIXED[5:] + (q, finv, rnd, None, fillers)


def seat_models(r: dict) -> list[str]:
    return [r["seat_models"][n] for n in r["events"]] if "seat_models" in r else [r["model"]] * r["n_agents"]


class Base:
    """Seekers and holders of a set of base runs, and which holder notes each seeker would have read."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.q = rows[0]["n_questions"]
        seekers, holders = [], []      # (run, model, q, reads) / (run, model, q, {title: first round carrying q})
        for ri, r in enumerate(rows):
            tr, ms = traces(r), seat_models(r)
            for ci, (holds, steps) in enumerate(tr):
                for q in range(self.q):
                    if q in holds:
                        first: dict[str, int] = {}
                        for s, (wt, carried, _) in enumerate(steps):
                            if wt and q in carried and wt not in first:
                                first[wt] = s
                        holders.append((ri, ms[ci], q, first))
                    elif any(q in h for k, (h, _) in enumerate(tr) if k != ci):
                        seekers.append((ri, ms[ci], q, [steps[s][2] for s in range(q + 1)]))
        self.n_seekers, self.n_holders = len(seekers), len(holders)
        self.uptake = self._uptake()
        self.cover = {}
        for q in range(self.q):
            for a in sorted({s[1] for s in seekers if s[2] == q}):
                for m in sorted({h[1] for h in holders if h[2] == q}):
                    S = [s for s in seekers if s[2] == q and s[1] == a]
                    H = [h for h in holders if h[2] == q and h[1] == m]
                    first = defaultdict(lambda: np.full(len(H), LATE, dtype=np.int16))
                    for y, h in enumerate(H):
                        for t, s in h[3].items():
                            first[t][y] = s
                    M = np.zeros((len(S), len(H)), dtype=bool)
                    for x, s in enumerate(S):
                        for rnd, t in enumerate(s[3]):
                            if t is not None and t in first:
                                M[x] |= first[t] <= rnd
                    self.cover[(q, a, m)] = (np.array([s[0] for s in S]), np.array([h[0] for h in H]), M)

    def _uptake(self) -> dict[str, float]:
        seen, used = Counter(), Counter()
        for r in self.rows:
            ms = dict(zip(r["events"], seat_models(r)))
            for name, ag in r["agents"].items():
                for a in ag["answers"]:
                    if a["chance"] and a["seen"]:
                        seen[ms[name]] += 1
                        used[ms[name]] += a.get("used_chosen", a["used_other"])
        return {m: used[m] / seen[m] for m in seen}

    def pi(self, w: np.ndarray | None = None) -> dict[tuple, tuple[np.ndarray, np.ndarray]]:
        """(q, seeker model, holder model) -> (pi per seeker, seeker weight); holders of the seeker's own run excluded.
        w: how often each base run is drawn (a resample); None = once each."""
        w = np.ones(len(self.rows)) if w is None else w
        out = {}
        for k, (srun, hrun, M) in self.cover.items():
            hw = w[hrun][None, :] * (srun[:, None] != hrun[None, :])
            tot = hw.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = (np.where(tot > 0, (M * hw).sum(1) / np.maximum(tot, 1e-12), np.nan), w[srun])
        return out


def mean_pi(pis: dict) -> dict[tuple, float]:
    acc = defaultdict(lambda: np.zeros(2))
    for (_, a, m), (p, w) in pis.items():
        ok = ~np.isnan(p)
        acc[(a, m)] += [(p[ok] * w[ok]).sum(), w[ok].sum()]
    return {k: v[0] / v[1] for k, v in acc.items()}


def found(pis: dict, a: str, q: int, hm: dict[str, int], one: dict | None = None) -> float:
    """Chance a seeker of model a finds question q's value when hm[m] copies of model m hold it."""
    ms = [m for m in hm if hm[m] > 0]
    if one is not None:
        return float(1 - np.prod([(1 - one[(a, m)]) ** hm[m] for m in ms]))
    p0, w = pis[(q, a, ms[0])]
    miss = np.ones_like(p0)
    for m in ms:
        miss = miss * (1 - np.nan_to_num(pis[(q, a, m)][0])) ** hm[m]
    ok = ~np.isnan(p0)
    return float(((1 - miss) * w)[ok].sum() / w[ok].sum())


def target_chances(rows: list[dict]) -> list[tuple[str, int, tuple]]:
    """Every chance of a target cell: (seeker model, question index, holder counts by model)."""
    out = []
    for r in rows:
        ms, names, qs = seat_models(r), list(r["events"]), r["questions"]
        holds = [set(r["holds"][n]) for n in names]
        for ci, n in enumerate(names):
            for a in r["agents"][n]["answers"]:
                if a["chance"]:
                    hm = Counter(ms[k] for k in range(len(names)) if k != ci and a["state"] in holds[k])
                    out.append((ms[ci], qs.index(a["state"]), tuple(sorted(hm.items()))))
    return out


def predict(base: Base, tch: list, w: np.ndarray | None = None) -> dict[str, float]:
    pis = base.pi(w)
    one = mean_pi(pis)
    cache: dict = {}
    tot = np.zeros(4)
    for a, q, hm in tch:
        if (a, q, hm) not in cache:
            f, o = found(pis, a, q, dict(hm)), found(pis, a, q, dict(hm), one)
            # NOTE (24 Nov 2026, HK Tan): was 0.0. No value ever seen in the base runs means the use rate is
            # unmeasured (0/0), not zero; assume 1. Changes Qwen3-32B only. Details in law_wrong.py, WrongBase.
            u = base.uptake.get(a, 1.0)
            cache[(a, q, hm)] = np.array([f, o, f * u, o * u])
        tot += cache[(a, q, hm)]
    seen, seen_one, cr, cr_one = tot / len(tch)
    return {"seen": seen, "seen_one": seen_one, "cr": cr, "cr_one": cr_one}


def observed(rows: list[dict]) -> dict[str, float]:
    ch = [a for r in rows for a in chances(r)]
    return {"cr": sum(a.get("used_chosen", a["used_other"]) for a in ch) / len(ch),
            "seen": sum(a["seen"] for a in ch) / len(ch)}


def pct(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]


def family(name: str, base_rows: list[dict], targets: dict[str, list[dict]], reps: int, seed: int = 0) -> list[dict]:
    base = Base(base_rows)
    rng, prng = np.random.default_rng(seed), random.Random(seed)
    nb = len(base_rows)
    boots = [rng.multinomial(nb, [1 / nb] * nb).astype(float) for _ in range(reps)]
    out = []
    for label, rows in targets.items():
        if not rows:
            continue
        tch = target_chances(rows)
        pt, bs = predict(base, tch), [predict(base, tch, w) for w in boots]
        per = [(sum(a.get("used_chosen", a["used_other"]) for a in chances(r)), len(chances(r))) for r in rows]
        ob = []
        for _ in range(reps):
            s = [per[prng.randrange(len(per))] for _ in per]
            ob.append(sum(x for x, _ in s) / sum(y for _, y in s))
        hs = Counter(sum(c for _, c in hm) for _, _, hm in tch)
        out.append({"family": name, "target": label, "n_runs": len(rows), "holders": dict(sorted(hs.items())),
                    **{f"pred_{k}": v for k, v in pt.items()},
                    **{f"pred_{k}_ci": pct([b[k] for b in bs]) for k in ("cr", "cr_one", "seen")},
                    **{f"obs_{k}": v for k, v in observed(rows).items()}, "obs_cr_ci": pct(ob),
                    "base_runs": nb, "uptake": base.uptake})
    return out


def cells(rows: list[dict], model: str, cond: tuple, src: str | None = None) -> dict[tuple, list[dict]]:
    """(source, N) -> runs of one model in one condition."""
    g = defaultdict(list)
    for r in rows:
        if r["model"] == model and condition(r) == cond and (src is None or r["_src"] == src):
            g[(r["_src"], r["n_agents"])].append(r)
    return dict(sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0] != "results")))


def label(src: str, n: int, prefix: str = "") -> str:
    return f"{prefix}N={n}" + ("" if src == "results" else f" ({src})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--extra", nargs="*", default=[], help="more result folders (e.g. ../wiki_game/data/swarm4round)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="title_law.json", help="file name inside the results directory")
    a = ap.parse_args()
    res = Path(a.results)
    rows = load(res, [Path(e) for e in a.extra])
    out = []
    q8 = "qwen3-8b"

    def run(name, base_rows, targets):
        if base_rows:
            out.extend(family(name, base_rows, targets, a.reps))
            print(f"  {name}: {len(base_rows)} base runs", flush=True)

    c8 = cells(rows, q8, fixed())
    run("Qwen3-8B, 2 answers each (base: pairs)", c8.get(("results", 2), []),
        {label(s, n): g for (s, n), g in c8.items()}
        | {label(s, n, "random split "): g for (s, n), g in cells(rows, q8, fixed(rnd=1)).items()}
        | {label(s, n, "1 answer each "): g for (s, n), g in cells(rows, q8, fixed(finv=4)).items()})
    c1 = cells(rows, q8, fixed(finv=4))
    run("Qwen3-8B, 1 answer each (base: 4 copies)", c1.get(("results", 4), []), {label(s, n): g for (s, n), g in c1.items()})
    c8q = cells(rows, q8, fixed(q=8))
    run("Qwen3-8B, 8 questions (base: pairs)", c8q.get(("results", 2), []), {label(s, n): g for (s, n), g in c8q.items()})
    # 8 questions with fewer answers each (scripts/run_split_grid.sh): each from its own smallest runs, and the
    # 8-question pairs' law applied to them (does the share of the quiz a copy holds change how it picks titles?)
    for finv, each in ((4, "2 answers each"), (8, "1 answer each")):
        cs = cells(rows, q8, fixed(q=8, finv=finv))
        run(f"Qwen3-8B, 8 questions, {each} (base: {finv} copies)", cs.get(("results", finv), []),
            {label(s, n): g for (s, n), g in cs.items()})
        run(f"Qwen3-8B, 8 questions, {each} (base: 8-question pairs)", c8q.get(("results", 2), []) if cs else [],
            {label(s, n): g for (s, n), g in cs.items()})
    cg = cells(rows, "gemma-3-27b", fixed())
    run("Gemma 3 27B (base: pairs)", cg.get(("results", 2), []), {label(s, n): g for (s, n), g in cg.items()})
    mixes = sorted({r["model"] for r in rows if "seat_models" in r and condition(r) == fixed()})
    run("Mixed Qwen3-8B + Gemma 3 27B (base: both models' pairs)",
        c8.get(("results", 2), []) + cg.get(("results", 2), []),
        {m: [r for r in rows if r["model"] == m and condition(r) == fixed()] for m in mixes})
    for twin, txt in ((0, "aware"), (1, "twin")):
        ca = cells(rows, q8, fixed(aw="aware", twin=twin), "results")
        run(f"Qwen3-8B, {txt} text (base: 4 copies)", ca.get(("results", 4), []),
            {label(s, n): g for (s, n), g in ca.items() if n >= 4})
    for m, name in (("qwen3-4b", "Qwen3-4B"), ("qwen3-14b", "Qwen3-14B"), ("qwen3-32b", "Qwen3-32B")):
        cm = cells(rows, m, fixed())
        base = next((g for (s, n), g in cm.items() if n == 2), [])
        run(f"{name} (base: pairs)", base, {label(s, n): g for (s, n), g in cm.items()})
    (res / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {res / a.out}\n")

    def f(v, ci):
        return f"{100 * v:5.1f} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]"
    print(f"{'family':48s} {'target':30s} {'holders':>8s}  {'measured':>18s}  {'title law':>18s}   {'one-number law':>18s}")
    for x in out:
        h = list(x["holders"])
        hs = h[0] if len(h) == 1 else f"{h[0]}-{h[-1]}"
        ok = " " if x["pred_cr_ci"][0] <= x["obs_cr"] <= x["pred_cr_ci"][1] else "*"
        print(f"{x['family'][:48]:48s} {x['target'][:30]:30s} {hs:>8}  {f(x['obs_cr'], x['obs_cr_ci'])}  "
              f"{f(x['pred_cr'], x['pred_cr_ci'])} {ok} {f(x['pred_cr_one'], x['pred_cr_one_ci'])}")
    print("\n* measured outside the title law's interval")


if __name__ == "__main__":
    main()
