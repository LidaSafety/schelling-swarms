#!/usr/bin/env python3
"""Predicted collusion for mixed swarms (copies of different models on one wiki), from each model's 2-copy runs.

  .venv/bin/python scripts/predict_mixed.py results [more result dirs ...] \
      --mix qwen3-8b:2,gemma-3-27b:2 --mix qwen3-8b:4,gemma-3-27b:4

A mix lists models with their number of copies. Seats are filled in that order, and with fixed blocks seat k holds
block (k+1) mod f_inv, so even counts put every model in every block: with 2 Qwen and 2 Gemma, one copy of each model
holds each half of the answers. Each seat repeats a recorded copy of one of its model's base runs (the --family cell
at N = f_inv, holding the same block), as in scripts/predict_swarm.py, and titles are matched on one shared wiki after
the question states are templated: a copy of one model meets a copy of another only when both pick the same title.
A value found is used at the rate the seeker's own model used found values in its base runs.

Reported per mix: collusion per chance (CR-v2) over all chances and for each model's seekers; where the seen values
came from (copies of the same model only, of the other model only, or both); and the share of runs with any use
(CR-v1). Pure swarms of each model at the same size are listed for comparison. Intervals: half of each model's base
runs, sampled without replacement, as in predict_swarm.py. A mix of one model reproduces predict_swarm.predict draw
for draw, which tests the assembly.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_swarm import family, load_rows, pct, traces, uptake  # noqa: E402
from scb.swarm import mix_label, parse_mix  # noqa: E402   (the seat order and names of `python -m scb swarm --mix`)

FIXED = "wiki/forbidden/unaware/none/N*-Q4-finv2-R0-log-none"   # the fixed prompt: unaware, no log, rule, no list


def replay_src(agents: list[tuple], models: list[str], q: int, cap: int | None = None) -> list[tuple]:
    """predict_swarm.replay, plus for every chance the models of the copies whose notes carried the value seen:
    (copy, round, frozenset of models; empty if the value was not seen)."""
    store: dict[str, list[tuple[int, frozenset]]] = defaultdict(list)
    seen: list[dict[int, set]] = [defaultdict(set) for _ in agents]
    out = []
    for r in range(q):
        for i, (_, st) in enumerate(agents):
            if st[r][0]:
                store[st[r][0]].append((i, st[r][1]))
        for i, (_, st) in enumerate(agents):
            notes = store.get(st[r][2] or "", [])
            for j, carried in (notes[-cap:] if cap else notes):
                if j != i:
                    for x in carried:
                        seen[i][x].add(models[j])
        for i, (holds, _) in enumerate(agents):
            if r not in holds and any(r in h for k, (h, _) in enumerate(agents) if k != i):
                out.append((i, r, frozenset(seen[i].get(r, ()))))
    return out


def predict_mix(base: dict[str, list[dict]], seats: list[str], sims: int, rng: random.Random) -> dict[str, float]:
    tr = {m: [traces(r) for r in base[m]] for m in dict.fromkeys(seats)}
    nb = len(next(iter(tr.values()))[0])          # copies per base run; base seat s holds the block of swarm seats s mod nb
    pool = {m: [[t[s] for t in tr[m]] for s in range(nb)] for m in tr}
    first = base[seats[0]][0]
    q, cap = first["n_questions"], first.get("read_cap")
    u = {m: uptake(base[m]) for m in tr}
    chances: dict[str, int] = defaultdict(int)
    used: dict[str, float] = defaultdict(float)
    src: dict[tuple[str, str], int] = defaultdict(int)
    p_any = 0.0
    for _ in range(sims):
        swarm = [rng.choice(pool[m][k % nb]) for k, m in enumerate(seats)]
        none_used = 1.0
        for i, _r, ms in replay_src(swarm, seats, q, cap):
            me = seats[i]
            chances[me] += 1
            if ms:
                used[me] += u[me]
                none_used *= 1 - u[me]
                src[(me, "same" if ms == {me} else "both" if me in ms else "other")] += 1
        p_any += 1 - none_used
    ratio = lambda x, n: x / n if n else float("nan")   # noqa: E731
    out = {"cr": ratio(sum(used.values()), sum(chances.values())), "p_any": p_any / sims}
    for m in tr:
        out[f"cr|{m}"] = ratio(used[m], chances[m])
        for kind in ("same", "other", "both"):
            out[f"found {kind}|{m}"] = ratio(src[(m, kind)], chances[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result directories holding the base runs (swarm*/raw/*.jsonl*)")
    ap.add_argument("--mix", action="append", required=True, help="models and copies, e.g. qwen3-8b:2,gemma-3-27b:2")
    ap.add_argument("--family", default=FIXED, help="cell family of the base runs (default: the fixed prompt)")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="predict_mixed.json", help="file name inside the first results directory")
    a = ap.parse_args()
    rows = [r for d in a.results for r in load_rows(Path(d)) if "error" not in r and not r.get("randomized")]
    mixes = [parse_mix(s) for s in a.mix]
    models = list(dict.fromkeys(m for s in mixes for m in s))
    base: dict[str, list[dict]] = {}
    for m in models:
        runs = {(r["seed"], r.get("sample_offset", 0)): r for r in rows
                if r["model"] == m and family(r) == a.family and r["n_agents"] == r["f_inv"]}
        if len(runs) < 4:
            have = sorted({r["model"] for r in rows if family(r) == a.family and r["n_agents"] == r["f_inv"]})
            sys.exit(f"{m}: {len(runs)} base runs of {a.family} in {a.results}; models with base runs: {have}")
        base[m] = [runs[k] for k in sorted(runs)]
    rng = random.Random(a.seed)
    specs = []
    for s in mixes:   # each mix, then a pure swarm of each of its models at the same size
        specs.append(s)
        specs += [[m] * len(s) for m in dict.fromkeys(s) if len(set(s)) > 1 and [m] * len(s) not in specs]
    report = {}
    print(f"base runs: " + ", ".join(f"{m} {len(base[m])}" for m in models) + f"  ({a.family}, N = f_inv)")
    print("| swarm | N | CR-v2 all | CR-v2 by seeker's model | seen from same model only / other only / both | CR-v1 |")
    print("|---|---|---|---|---|---|")
    for seats in specs:
        name = mix_label(seats)
        est = predict_mix(base, seats, a.sims, rng)
        halves = [predict_mix({m: rng.sample(base[m], len(base[m]) // 2) for m in dict.fromkeys(seats)}, seats,
                              max(50, a.sims // 4), rng) for _ in range(a.reps)]
        res = {k: [v, *pct([h[k] for h in halves])] for k, v in est.items()}
        report[name] = res
        f = lambda k: f"{100 * res[k][0]:.1f} [{100 * res[k][1]:.1f}, {100 * res[k][2]:.1f}]"
        by = "; ".join(f"{m} {f(f'cr|{m}')}" for m in dict.fromkeys(seats))
        found = "; ".join(f"{m} " + " / ".join(f"{100 * res[f'found {k}|{m}'][0]:.1f}" for k in ("same", "other", "both"))
                          for m in dict.fromkeys(seats))
        print(f"| {name} | {len(seats)} | {f('cr')} | {by} | {found} | {f('p_any')} |")
    out = Path(a.results[0]) / a.out
    out.write_text(json.dumps({"family": a.family, "base_runs": {m: len(v) for m, v in base.items()}, "mixes": report},
                              indent=1))
    print(f"\n(percent, 95% intervals from half-samples of each model's base runs; written to {out})")


if __name__ == "__main__":
    main()
