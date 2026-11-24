#!/usr/bin/env python3
"""Scaling predictor for the N-agent dead drop (scb/swarm.py): what N=2 behaviour implies for a swarm of N copies.

  .venv/bin/python scripts/predict_swarm.py [results_dir] [--sizes 2 4 8 16 32] [--models qwen3-8b]

For every cell family with base runs (fixed blocks, R=0; the base is the smallest swarm in which every question has
a holder: N = f_inv, so pairs for f_inv=2 and N=4 for f_inv=4), swarms of N copies are assembled from the recorded
copies of independent base runs (row["events"]): each copy repeats what it did in its pair, round by round (the title it wrote,
which question values its note carried, the title it read). Seat k of the swarm draws from seat k mod f_inv of the base runs,
which holds the same block, as fixed blocks seat them. Titles are compared after replacing each run's question states by
their round ({Q1}..{QQ}, case kept), and a value found is used at the pairs' uptake (used / seen over chances).

The prediction assumes copies in a swarm act exactly as in a pair: no reaction to crowded pages, and no stated head
count (the aware wording says "{k} other instances" at N>2; awareness=unaware says nothing). Observed N>2 cells of
the same family are shown next to it: observed - predicted is what the swarm adds beyond more chances to meet.
Intervals: observed resamples runs; predicted resamples half of the N=2 seeds without replacement (balanced
half-sampling: duplicated runs would put identical copies in a swarm and bias coverage down). Twin contrasts
(twin wording minus aware wording) use the same half of the seeds for both arms.

Rows saved before commit 107ae64 (fields hint 0/1/2 and twin 0/1 instead of log none/A/B) are read too:
normalize() gives them the current field. The twin arm ("copies of you", --twin 1) exists only in those old rows; its
families end in "_twin", e.g. wiki/forbidden/aware/none/N*-Q4-finv2-R0-log-none_twin.

Self-check: replaying each N=2 run on its own must reproduce what its copies saw (mismatches are reported per family).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scb.io import read_jsonl  # noqa: E402
from scb.swarm import cell_name  # noqa: E402


LOG_OF_HINT = {0: "none", 1: "A", 2: "B"}   # rows saved before 107ae64: hint -> log


def normalize(r: dict) -> dict:
    """Give a row saved before commit 107ae64 (fields hint 0/1/2 and twin 0/1) the current field `log`
    (none/A/B), in place; rows from the current code get twin=0. Returns the row."""
    if "log" not in r:
        r["log"] = LOG_OF_HINT[r.get("hint", 0)]
    r.setdefault("twin", 0)
    return r


# The planted-notes runs (scripts/run_plant_pairs.sh) are their own experiment: their arm without a plant repeats the
# fixed prompt, and their planted arms differ from it only by notes no condition records. Left out of every analysis
# that selects runs by condition; scripts/analyze_plant.py reads them.
SKIP_TAGS = ("swarm_plant",)
# Tags added after results/predict_swarm.json was written (8 November). This script draws every family from one
# random stream, so their families would shift the draws of the others; by default it reads only the runs it was
# written from (--all-tags reads every folder but the planted notes).
LATER_TAGS = ("swarm_claude_pairs", "swarm_comp_n4_gemma", "swarm_deepseek", "swarm_fill", "swarm_or_pairs",
              "swarm_q8f", "swarm_rulehalves", "swarm_ruleoff", "swarm_scale", "swarm_twin")


def load_rows(res: Path, pattern: str = "swarm*/raw/*.jsonl*", skip: tuple[str, ...] = SKIP_TAGS) -> list[dict]:
    """Every row under `res` whose file matches `pattern`, old or current format, normalized (folders starting with
    `skip` left out)."""
    return [normalize(r) for f in sorted(res.glob(pattern)) if not f.parts[-3].startswith(skip) for r in read_jsonl(f)]


def family(r: dict) -> str:
    """The cell name with N replaced by *. Old twin=1 rows get "_twin" after the log part: without it they would share
    a name with the aware cells, and with it the families sort as the old names did (twin after aware), so a given
    --seed gives the same predictions as before the rename."""
    # the runner marks twin cells with its own "/twin" (patches/twin-text-runner.patch): this file keeps "_twin"
    name = re.sub(r"/N\d+-", "/N*-", cell_name(normalize(r)).removesuffix("/twin"))
    return re.sub(r"(-log-[^/]+)", r"\1_twin", name, count=1) if r["twin"] else name


def template(name: str | None, qs: list[str]) -> str | None:
    if not name:
        return name
    for j, s in enumerate(qs, 1):
        name = name.replace(s, f"{{Q{j}}}").replace(s.lower(), f"{{q{j}}}").replace(s.upper(), f"{{Q{j}!}}")
    return name


def traces(r: dict) -> list[tuple[frozenset, list[tuple]]]:
    """Per copy, in seat order: (question indices it holds, [(write title, indices carried, read title) per round])."""
    qs = r["questions"]
    ix = {s: j for j, s in enumerate(qs)}
    out = []
    for name in r["events"]:
        ev = r["events"][name]
        w = {e["round"]: e for e in ev["writes"]}
        rd = {e["round"]: e for e in ev["reads"]}
        steps = [(template(w[k]["path"], qs) if k in w else None,
                  frozenset(ix[s] for s in w[k]["posts"] if s in ix) if k in w else frozenset(),
                  template(rd[k]["path"], qs) if k in rd else None) for k in range(1, r["n_questions"] + 1)]
        out.append((frozenset(ix[s] for s in r["holds"][name]), steps))
    return out


def replay(agents: list[tuple[frozenset, list[tuple]]], q: int, cap: int | None = None) -> list[tuple[int, int, bool]]:
    """Play recorded choices on one shared channel; (copy, round, saw the value) for every chance (a question the copy
    does not hold and another copy does). Notes land in seat order; a read sees the newest `cap` notes if capped."""
    store: dict[str, list[tuple[int, frozenset]]] = defaultdict(list)
    seen: list[set] = [set() for _ in agents]
    out = []
    for r in range(q):
        for i, (_, st) in enumerate(agents):
            if st[r][0]:
                store[st[r][0]].append((i, st[r][1]))
        for i, (_, st) in enumerate(agents):
            notes = store.get(st[r][2] or "", [])
            for j, carried in (notes[-cap:] if cap else notes):
                if j != i:
                    seen[i] |= carried
        for i, (holds, _) in enumerate(agents):
            if r not in holds and any(r in h for k, (h, _) in enumerate(agents) if k != i):
                out.append((i, r, r in seen[i]))
    return out


def chances(r: dict) -> list[dict]:
    return [a for ag in r["agents"].values() for a in ag["answers"] if a["chance"]]


def uptake(rows: list[dict]) -> float:
    ch = [a for r in rows for a in chances(r)]
    s = sum(a["seen"] for a in ch)
    return sum(a.get("used_chosen", a["used_other"]) for a in ch) / s if s else 0.0


def self_check(pairs: list[dict]) -> int:
    bad = 0
    for r in pairs:
        got = {(i, k): f for i, k, f in replay(traces(r), r["n_questions"], r.get("read_cap"))}
        want = {(i, a["round"] - 1): a["seen"] for i, ag in enumerate(r["agents"].values())
                for a in ag["answers"] if a["chance"]}
        bad += got != want
    return bad


def predict(pairs: list[dict], n: int, sims: int, rng: random.Random) -> dict[str, float]:
    tr = [traces(r) for r in pairs]
    nb = len(tr[0])     # copies per base run; base seat s holds block (s+1) mod f_inv, as do swarm seats k = s mod nb
    seats = [[t[s] for t in tr] for s in range(nb)]
    q, cap, u = pairs[0]["n_questions"], pairs[0].get("read_cap"), uptake(pairs)
    found = opps = 0
    p_any = 0.0
    for _ in range(sims):
        swarm = [rng.choice(seats[k % nb]) for k in range(n)]
        res = replay(swarm, q, cap)
        hits = sum(x[2] for x in res)
        found, opps = found + hits, opps + len(res)
        p_any += 1 - (1 - u) ** hits
    return {"cr": found / opps * u, "found": found / opps, "p_any": p_any / sims, "uptake": u}


def observed(rows: list[dict]) -> dict[str, float]:
    ch = [a for r in rows for a in chances(r)]
    used = sum(a.get("used_chosen", a["used_other"]) for a in ch)
    return {"cr": used / len(ch), "found": sum(a["seen"] for a in ch) / len(ch),
            "p_any": sum(any(a.get("used_chosen", a["used_other"]) for a in chances(r)) for r in rows) / len(rows)}


def pct(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]] if xs else [float("nan")] * 2


def fmt(v: list[float] | None) -> str:
    return f"{100 * v[0]:5.1f} [{100 * v[1]:4.1f}, {100 * v[2]:4.1f}]" if v else "  –"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--sims", type=int, default=400, help="assembled swarms per estimate")
    ap.add_argument("--reps", type=int, default=100, help="half-samples / bootstrap resamples for intervals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aware-base", type=int, default=0,
                    help="base size for aware families (e.g. 4: the aware text says 'Another instance' at N=2 but "
                         "'{k} other instances' from N=4 on, and the wording alone moves titles); default: N = f_inv")
    ap.add_argument("--out", default="predict_swarm.json", help="file name inside the results directory")
    ap.add_argument("--all-tags", action="store_true", help="also read the tags added after 8 November (LATER_TAGS)")
    ap.add_argument("--models", nargs="+", help="only these models (default: all). Families are drawn in sorted "
                                                 "order from one random stream, so other models' runs shift the draws")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    res = Path(a.results)
    rows = [r for r in load_rows(res, skip=SKIP_TAGS + (() if a.all_tags else LATER_TAGS)) if "error" not in r and (not a.models or r["model"] in a.models)]
    # (model, family) -> N -> (seed, sample_offset) -> run
    fams: dict[tuple[str, str], dict[int, dict[tuple, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if not r.get("randomized"):
            fams[(r["model"], family(r))][r["n_agents"]][(r["seed"], r.get("sample_offset", 0))] = r
    report: dict[str, dict] = {}
    for (model, fam), by_n in sorted(fams.items()):
        f = next(iter(next(iter(by_n.values())).values()))["f_inv"]
        b = a.aware_base if a.aware_base and "/aware/" in fam else f
        pairs_by_seed = by_n.get(b, {})
        if len(pairs_by_seed) < 4:
            continue
        keys = sorted(pairs_by_seed)
        pairs = [pairs_by_seed[k] for k in keys]
        print(f"\n## {model}  {fam}   ({len(pairs)} base runs at N={b}; self-check {self_check(pairs)} mismatches)")
        print("| N | predicted CR | observed CR | predicted found | observed found | predicted P_any | observed P_any |")
        print("|---|---|---|---|---|---|---|")
        fam_out: dict[str, dict] = {}
        for n in [n for n in a.sizes if n >= b]:
            est = predict(pairs, n, a.sims, rng)
            halves = [predict(rng.sample(pairs, len(pairs) // 2), n, max(50, a.sims // 4), rng) for _ in range(a.reps)]
            pr = {k: [est[k], *pct([h[k] for h in halves])] for k in ("cr", "found", "p_any")}
            ob = None
            if n in by_n and len(by_n[n]) >= 2:
                obs_rows = list(by_n[n].values())
                o = observed(obs_rows)
                boots = [observed([rng.choice(obs_rows) for _ in obs_rows]) for _ in range(a.reps)]
                ob = {k: [o[k], *pct([b[k] for b in boots])] for k in o}
            fam_out[str(n)] = {"predicted": pr, "observed": ob, "runs": len(by_n.get(n, {})), "uptake": est["uptake"]}
            print(f"| {n} | {fmt(pr['cr'])} | {fmt(ob and ob['cr'])} | {fmt(pr['found'])} | {fmt(ob and ob['found'])} "
                  f"| {fmt(pr['p_any'])} | {fmt(ob and ob['p_any'])} |")
        report[f"{model} {fam}"] = fam_out
    # twin contrast of the prediction (old rows only): twin minus aware wording, on the same half of the seeds for both
    print("\n## predicted twin effect, twin minus aware wording (percentage points of CR; paired half-samples of seeds)")
    con: dict[str, dict] = {}
    for (model, fam), by_n in sorted(fams.items()):
        if "_twin" not in fam:
            continue
        base = (model, fam.replace("_twin", "", 1))
        if base not in fams:
            continue
        f = a.aware_base or int(fam.split("-finv")[1].split("-")[0])
        p1, p0 = fams[(model, fam)].get(f, {}), fams[base].get(f, {})
        seeds = sorted(set(p1) & set(p0))
        if len(seeds) < 4:
            continue
        line = []
        con_f: dict[str, list[float]] = {}
        for n in [n for n in a.sizes if n >= f]:
            d = predict([p1[s] for s in seeds], n, a.sims, rng)["cr"] - predict([p0[s] for s in seeds], n, a.sims, rng)["cr"]
            hs = []
            for _ in range(a.reps):
                h = rng.sample(seeds, len(seeds) // 2)
                hs.append(predict([p1[s] for s in h], n, max(50, a.sims // 4), rng)["cr"]
                          - predict([p0[s] for s in h], n, max(50, a.sims // 4), rng)["cr"])
            con_f[str(n)] = [d, *pct(hs)]
            line.append(f"N={n}: {fmt(con_f[str(n)])}")
        con[f"{model} {fam}"] = con_f
        print(f"- {model} {base[1]}, twin minus aware ({len(seeds)} paired seeds): " + "; ".join(line))
    out = res / a.out
    out.write_text(json.dumps({"families": report, "twin_contrast": con}, indent=1))
    print(f"\n(percent, 95% intervals; written to {out})")


if __name__ == "__main__":
    main()
