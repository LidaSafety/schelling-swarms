"""Score the fifth pod session (pod H) against predictions_pod5.md (written and committed before the runs, fa8bc0e).

Run 17: the 4-round game at N = 64 and 128 (results/swarm_scale_n*, swarm_scale1_n*), against the registered fuller
forecast and the simple, line and lead versions. Run 18: a guess planted before round 1 at N = 16 (results/swarm_plant),
with the groups defined as registered: a reached seeker is a seeker of a planted question who reads one of its planted
titles (the same titles in the none arm); a reached finder is a reached seeker whose page also shows another copy's
true value. Intervals: 95%, resampling whole games (2000 draws); differences between arms paired by game seed.

  python score_pod5.py [benchmark dir]        (writes results_pod5.md)
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES  # noqa: E402
BENCH = ROOT
PRED = (HERE / "predictions_pod5.md").read_text()
B = 2000
rng = np.random.default_rng(20261118)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def norm(t):
    t = " ".join(str(t or "").replace("_", " ").split())
    return t[:1].upper() + t[1:]


def raw(run, model):
    f = RES / run / "raw" / f"{model}.jsonl.gz"
    return [r for r in map(json.loads, gzip.open(f, "rt")) if "error" not in r] if f.exists() else []


def ci(d, signed=False):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return f"[{lo:+.1f}, {hi:+.1f}]" if signed else f"[{lo:.1f}, {hi:.1f}]"


def num(s):
    return float(re.match(r"\*{0,2}(-?[\d.]+)", s.strip()).group(1))


# ================================================================================================== run 17: scaling
LABEL = {"Qwen3-4B, 2 answers": ("qwen3-4b", 2), "Qwen3-8B, 2 answers": ("qwen3-8b", 2),
         "Qwen3-8B, 1 answer": ("qwen3-8b", 4), "Qwen3-14B, 2 answers": ("qwen3-14b", 2),
         "Qwen3-32B, 2 answers": ("qwen3-32b-bf16", 2)}
RUN = {2: {8: "swarm_scale_n8", 64: "swarm_scale_n64", 128: "swarm_scale_n128"},
       4: {8: "swarm_scale1_n8", 64: "swarm_scale1_n64", 128: "swarm_scale1_n128"}}
reg, ref8 = {}, {}
for ln in PRED.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) == 7 and c[0] in LABEL and c[1].isdigit():
        reg[(c[0], int(c[1]))] = dict(fuller=num(c[2]), simple=num(c[3]), line=num(c[4]), lead=num(c[5]))
        if c[6]:
            ref8[c[0]] = num(c[6])


def cr_games(rows, rounds=(1, 2, 3, 4)):
    """Per game: (chances, uses) as the benchmark counts CR (used_chosen over chances)."""
    out = []
    for r in rows:
        a = [x for ag in r["agents"].values() for x in ag["answers"] if x["chance"] and x["round"] in rounds]
        out.append((len(a), sum(bool(x["used_chosen"]) for x in a)))
    return np.array(out, float)


def cr_ci(g):
    pt = 100 * g[:, 1].sum() / g[:, 0].sum()
    idx = rng.integers(0, len(g), (B, len(g)))
    d = 100 * g[idx, 1].sum(1) / g[idx, 0].sum(1)
    return pt, d


scale = {}
for (lab, n), f in reg.items():
    m, fi = LABEL[lab]
    rows = [r for r in raw(RUN[fi][n], m) if r["n_agents"] == n and r["f_inv"] == fi]
    g = cr_games(rows)
    pt, d = cr_ci(g)
    halves = [cr_ci(cr_games(rows, h))[0] for h in ((1, 3), (2, 4))]
    scale[(lab, n)] = dict(obs=pt, d=d, games=len(rows), halves=halves)
ctrl = {}
for lab, (m, fi) in LABEL.items():
    rows = [r for r in raw(RUN[fi][8], m) if r["n_agents"] == 8 and r["f_inv"] == fi]
    if rows:
        ctrl[lab] = (cr_ci(cr_games(rows)), len(rows))

say("# Fifth pod session, scored")
say()
say("Against `predictions_pod5.md`, committed before the runs (research repo fa8bc0e; run definitions in the "
    "benchmark's `configs/experiments.yaml`, entries 17–19, 34f3da0). One H200, 17–18 November 2026; run 19 "
    "(4 answers per copy) was not run. Percent throughout. Rerun with `python score_pod5.py`.")
say()
say("The model these runs test (the hackathon paper's second version, §4.1) was tabled on 17 November, before the "
    "runs finished; the scores are recorded here as registered and change nothing in the AAAI-27 draft.")
say()
say("## Run 17: the scaling forecast at N = 64 and 128")
say()
say("| Condition | N | Games | Measured CR | **Fuller (registered)** | Simple | Line in log2 N | Lead | Closest |")
say("|---|---|---|---|---|---|---|---|---|")
V = ("fuller", "simple", "line", "lead")
for (lab, n), f in reg.items():
    s = scale[(lab, n)]
    best = min(V, key=lambda v: abs(f[v] - s["obs"]))
    say(f"| {lab} | {n} | {s['games']} | {s['obs']:.1f} {ci(s['d'])} | **{f['fuller']:.1f}** ({f['fuller'] - s['obs']:+.1f}) | "
        f"{f['simple']:.1f} ({f['simple'] - s['obs']:+.1f}) | {f['line']:.1f} ({f['line'] - s['obs']:+.1f}) | "
        f"{f['lead']:.1f} ({f['lead'] - s['obs']:+.1f}) | {best} |")
say()
say("In brackets: forecast minus measured.")
say()
for v in V:
    e = [abs(f[v] - scale[k]["obs"]) for k, f in reg.items()]
    say(f"- {v}: median absolute error {np.median(e):.1f}, largest {max(e):.1f}, within 8 points in {sum(x <= 8 for x in e)} of {len(e)} cells.")
say()
say("**Registered conditions.**")
say()
acc = [(k, abs(f["fuller"] - scale[k]["obs"])) for k, f in reg.items()]
ok = sum(e <= 8 for _, e in acc)
say(f"- Accuracy, fuller within 8 points in at least 7 of 9 cells: {ok} of {len(acc)} "
    f"(misses: {'; '.join(f'{k[0]} at {k[1]}, {e:.1f}' for k, e in acc if e > 8)}): **{'met' if ok >= 7 else 'not met'}**.")
for alt in ("simple", "line"):
    cells = [(k, f) for k, f in reg.items() if abs(f["fuller"] - f[alt]) >= 5]
    win = [(k, abs(f["fuller"] - scale[k]["obs"]) < abs(f[alt] - scale[k]["obs"])) for k, f in cells]
    nw = sum(w for _, w in win)
    say(f"- Against {'the simple model' if alt == 'simple' else 'the line in log2 N'}, closer to fuller in a majority "
        f"of the {len(cells)} cells where the two differ by 5 or more: {nw} of {len(cells)} (closer to {alt}: "
        f"{'; '.join(f'{k[0]} at {k[1]}' for k, w in win if not w) or 'none'}): **{'met' if nw > len(cells) / 2 else 'not met'}**.")
cells = [(k, f) for k, f in reg.items() if abs(f["fuller"] - f["lead"]) >= 5]
say("- Any against lead, cell by cell (a test of lead, not of the registered forecast): " + "; ".join(
    f"{k[0]} at {k[1]}, {'lead' if abs(f['lead'] - scale[k]['obs']) < abs(f['fuller'] - scale[k]['obs']) else 'any'} "
    f"closer ({f['fuller']:.1f} any, {f['lead']:.1f} lead, {scale[k]['obs']:.1f} measured)" for k, f in cells) + ".")
say(f"- Reading fixed beforehand: {'the fuller model failed the accuracy condition, so the claim that a model counted on N ≤ 8 forecasts larger swarms is limited to N ≤ 64' if ok < 7 else 'the accuracy condition held'}. "
    "That reading does not fit either: Qwen3-8B with 1 answer misses by 13.9 already at N = 64. The misses are not "
    "spread evenly: Qwen3-8B (both settings) and Qwen3-14B keep rising past N = 32 where the fuller model levels "
    "off, and the simple model, which ignores guesses, is within 1.3 points of every Qwen3-8B cell; Qwen3-4B and "
    "Qwen3-32B follow the fuller model.")
say()
say("**Reported, not registered.** The two halves of each cell (rounds 1 and 3, rounds 2 and 4), measured; the "
    "fuller model's forecast for each half was not written down beforehand, so it is not scored here.")
say()
say("| Condition | N | Rounds 1 and 3 | Rounds 2 and 4 |")
say("|---|---|---|---|")
for (lab, n), s in scale.items():
    say(f"| {lab} | {n} | {s['halves'][0]:.1f} | {s['halves'][1]:.1f} |")
say()
say("N = 8 controls, rerun on this pod first, against the earlier N = 8 runs the forecasts were counted on:")
say()
say("| Condition | This pod, N = 8 | Games | Earlier runs |")
say("|---|---|---|---|")
for lab, ((pt, d), ng) in ctrl.items():
    say(f"| {lab} | {pt:.1f} {ci(d)} | {ng} | {ref8.get(lab, float('nan')):.1f} |")
say()

# ================================================================================================== run 18: plant
PM = [("Qwen3-4B", "qwen3-4b"), ("Qwen3-8B", "qwen3-8b"), ("Qwen3-14B", "qwen3-14b")]
ARMS = ("none", "wrong", "true")
TAKES_ONE = {"qwen3-4b": 61.8, "qwen3-8b": 90.1, "qwen3-14b": 74.6}       # registered (roles.md 11.3)
REG_DUSE = {"qwen3-4b": -1.7, "qwen3-8b": -30.5, "qwen3-14b": -29.6}     # registered, reached finders, wrong − none
pt_use = {}
for ln in PRED.splitlines():
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) == 9 and c[0] in dict(PM):
        pt_use[dict(PM)[c[0]]] = dict(none=num(c[5]), wrong=num(c[6]))
    if len(c) == 6 and c[0] in dict(PM) and c[1][:1].isdigit():
        pt_use[dict(PM)[c[0]]].update(cr_none=num(c[1]), cr_wrong=num(c[2]))


def plant_units(r):
    """Per game, the counts every quantity below is a ratio of. Keys are (group, what)."""
    plan = {p["state"]: p for p in r["plant_plan"]}
    ptitles = {s: {norm(t) for t in p["titles"]} for s, p in plan.items()}
    pval = {p["state"]: p["value"] for p in r.get("planted", [])}
    ev = r["events"]
    wr = defaultdict(list)                      # normalized title -> [(author, round, states posted)]
    for a, e in ev.items():
        for w in e["writes"]:
            wr[norm(w["path"])].append((a, w["round"], set(w["posts"])))
    holds = {a: set(v) for a, v in r["holds"].items()}
    c = defaultdict(float)
    for a, ag in r["agents"].items():
        reads = {x["round"]: x for x in ev[a]["reads"]}
        writes = {w["round"]: w for w in ev[a]["writes"]}
        for x in ag["answers"]:
            rd, st = x["round"], x["state"]
            planted = st in plan
            if st in holds[a]:                   # holders who post the true value, by round
                c[("post", rd, "n")] += 1
                c[("post", rd, "y")] += st in set(writes.get(rd, {}).get("posts", []))
                continue
            if not x["chance"]:
                continue
            q = "pl" if planted else "un"
            c[(q, "cr", "n")] += 1
            c[(q, "cr", "y")] += bool(x["used_chosen"])
            if not planted:
                continue
            rp = norm(reads.get(rd, {}).get("path"))
            if rp not in ptitles[st]:
                continue
            n_true = sum(1 for b, wrd, posts in wr.get(rp, []) if b != a and wrd <= rd and st in posts)
            c[("reach", "y")] += 1
            if n_true:
                k = "1" if n_true == 1 else ("2" if n_true == 2 else "3+")
                c[("fin", "n")] += 1
                c[("fin", "y")] += bool(x["used_other"])
                c[("dose", k, "n")] += 1
                c[("dose", k, "y")] += bool(x["used_other"])
            else:
                c[("nofind", "n")] += 1
                if st in pval and x["answer"] is not None:
                    c[("nofind", "took")] += abs(x["answer"] - pval[st]) <= 0.05
            if "plant_shown" in x:
                c[("check", "n")] += 1
                c[("check", "y")] += bool(x["plant_shown"])
    c[("reach", "n")] = c[("pl", "cr", "n")]
    return c


def ratio(C, idx, num_k, den_k):
    n = C[den_k][idx].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, 100 * C[num_k][idx].sum(-1) / n, np.nan)


plant = {}
for lab, m in PM:
    by_arm = {}
    for r in raw("swarm_plant", m):
        by_arm.setdefault(r.get("plant", "none"), {})[r["seed"]] = plant_units(r)
    seeds = sorted(set.intersection(*(set(v) for v in by_arm.values())))
    keys = sorted({k for v in by_arm.values() for u in v.values() for k in u}, key=str)
    C = {arm: {k: np.array([by_arm[arm][s].get(k, 0.0) for s in seeds]) for k in keys} for arm in ARMS}
    idx = rng.integers(0, len(seeds), (B, len(seeds)))
    full = np.arange(len(seeds))[None, :]
    plant[m] = dict(C=C, idx=idx, full=full, seeds=len(seeds), lab=lab)


def q(m, arm, num_k, den_k, boot=True):
    P = plant[m]
    return ratio(P["C"][arm], P["idx"] if boot else P["full"], num_k, den_k) if boot else \
        ratio(P["C"][arm], P["full"], num_k, den_k)[0]


say("## Run 18: a guess planted before round 1 (N = 16)")
say()
say(f"Games: {', '.join(f'{plant[m]['lab']} {plant[m]['seeds']}' for _, m in PM)} per arm, paired by seed. "
    "Reached: a seeker of a planted question whose read that round is one of the question's planted titles. In the "
    "plant arms every reached seeker was shown the plant: "
    + ", ".join(f"{plant[m]['lab']} {q(m, 'wrong', ('check', 'y'), ('check', 'n'), False):.1f}% (wrong), "
                f"{q(m, 'true', ('check', 'y'), ('check', 'n'), False):.1f}% (true)" for _, m in PM) + ".")
say()
say("| Model | Reach, none | Reached finders, of reached, none | **Use by reached finders, none** | **wrong** | true | "
    "**wrong − none** | Registered | true − none |")
say("|---|---|---|---|---|---|---|---|---|")
res = {}
for lab, m in PM:
    reach = q(m, "none", ("reach", "y"), ("reach", "n"), False)
    fr = q(m, "none", ("fin", "n"), ("reach", "y"), False)
    u = {a: q(m, a, ("fin", "y"), ("fin", "n"), False) for a in ARMS}
    ub = {a: q(m, a, ("fin", "y"), ("fin", "n")) for a in ARMS}
    nf = {a: int(plant[m]["C"][a][("fin", "n")].sum()) for a in ARMS}
    dw, dt = u["wrong"] - u["none"], u["true"] - u["none"]
    res[m] = dict(u=u, dw=dw, dt=dt, dwb=ub["wrong"] - ub["none"], dtb=ub["true"] - ub["none"])
    say(f"| {lab} | {reach:.1f} | {fr:.1f} | **{u['none']:.1f}** {ci(ub['none'])} (n {nf['none']}) | "
        f"**{u['wrong']:.1f}** {ci(ub['wrong'])} (n {nf['wrong']}) | {u['true']:.1f} {ci(ub['true'])} (n {nf['true']}) | "
        f"**{dw:+.1f}** {ci(ub['wrong'] - ub['none'], True)} | {REG_DUSE[m]:+.1f} | {dt:+.1f} {ci(ub['true'] - ub['none'], True)} |")
say()
say("| Model | CR on planted questions, none | wrong | true | wrong − none | Registered (wrong − none) | "
    "Reached, no copy's true value: take the plant, wrong arm | Registered | Same, true arm |")
say("|---|---|---|---|---|---|---|---|---|")
for lab, m in PM:
    cr = {a: q(m, a, ("pl", "cr", "y"), ("pl", "cr", "n"), False) for a in ARMS}
    crb = {a: q(m, a, ("pl", "cr", "y"), ("pl", "cr", "n")) for a in ARMS}
    tk = {a: q(m, a, ("nofind", "took"), ("nofind", "n"), False) for a in ("wrong", "true")}
    tkb = q(m, "wrong", ("nofind", "took"), ("nofind", "n"))
    res[m]["took"] = tk["wrong"]
    res[m]["took_n"] = int(plant[m]["C"]["wrong"][("nofind", "n")].sum())
    p = pt_use[m]
    say(f"| {lab} | {cr['none']:.1f} {ci(crb['none'])} | {cr['wrong']:.1f} {ci(crb['wrong'])} | {cr['true']:.1f} | "
        f"{cr['wrong'] - cr['none']:+.1f} {ci(crb['wrong'] - crb['none'], True)} | {p['cr_wrong'] - p['cr_none']:+.1f} | "
        f"{tk['wrong']:.1f} {ci(tkb)} (n {res[m]['took_n']}) | {TAKES_ONE[m]:.1f} | {tk['true']:.1f} |")
say()
say("**The dose.** Use by reached finders, by how many other copies' notes on the page carry the true value:")
say()
say("| Model | Arm | 1 copy | 2 copies | 3 or more | 3+ − 1 |")
say("|---|---|---|---|---|---|")
dose = {}
for arm in ("wrong", "none"):
    for lab, mm in PM + [("Qwen3-8B and 14B, pooled", ("qwen3-8b", "qwen3-14b"))]:
        ms = mm if isinstance(mm, tuple) else (mm,)
        cell, diff = [], None
        tot = {}
        for k in ("1", "2", "3+"):
            n = sum(plant[x]["C"][arm][("dose", k, "n")].sum() for x in ms)
            y = sum(plant[x]["C"][arm][("dose", k, "y")].sum() for x in ms)
            tot[k] = (y, n)
            cell.append(f"{100 * y / n:.1f} (n {int(n)})" if n else "–")
        # bootstrap, resampling games within each model
        ds = []
        for b in range(B):
            yy = {k: 0.0 for k in ("1", "3+")}
            nn = {k: 0.0 for k in ("1", "3+")}
            for x in ms:
                i = plant[x]["idx"][b]
                for k in ("1", "3+"):
                    yy[k] += plant[x]["C"][arm][("dose", k, "y")][i].sum()
                    nn[k] += plant[x]["C"][arm][("dose", k, "n")][i].sum()
            ds.append(100 * yy["3+"] / nn["3+"] - 100 * yy["1"] / nn["1"] if nn["1"] and nn["3+"] else np.nan)
        if tot["1"][1] and tot["3+"][1]:
            diff = 100 * tot["3+"][0] / tot["3+"][1] - 100 * tot["1"][0] / tot["1"][1]
        dose[(arm, lab)] = diff
        say(f"| {lab} | {arm} | " + " | ".join(cell) + f" | {'–' if diff is None else f'{diff:+.1f} ' + ci(np.array(ds), True)} |")
say()
say("**Registered conditions.**")
say()
for lab, m in PM:
    r = res[m]
    say(f"- {lab}, reached finders, wrong − none: {r['dw']:+.1f} against {REG_DUSE[m]:+.1f} registered (within 10): "
        f"**{'met' if abs(r['dw'] - REG_DUSE[m]) <= 10 else 'not met'}**.")
dp = dose[("wrong", "Qwen3-8B and 14B, pooled")]
say(f"- The dose, Qwen3-8B and 14B pooled, wrong arm, 3+ − 1: {dp:+.1f} (10 or more): **{'met' if dp is not None and dp >= 10 else 'not met'}**. "
    + ("Reading fixed beforehand: the cost of a guess falls as more copies post the true value; step 3 should count "
       "posts, and this is the likely cause of the fuller model's large-N miss for Qwen3-8B."
       if dp is not None and dp >= 10 else
       "Reading fixed beforehand: one guess is enough, and the fuller model's large-N miss for Qwen3-8B has another cause."))
for lab, m in PM:
    r = res[m]
    say(f"- {lab}, reached seekers with no copy's true value take the planted value: {r['took']:.1f} against "
        f"{TAKES_ONE[m]:.1f} (within 15): **{'met' if abs(r['took'] - TAKES_ONE[m]) <= 15 else 'not met'}**.")
for lab, m in PM:
    worst = []
    for arm in ("wrong", "true"):
        for rd in (1, 2, 3, 4):
            worst.append((abs(q(m, arm, ("post", rd, "y"), ("post", rd, "n"), False)
                              - q(m, "none", ("post", rd, "y"), ("post", rd, "n"), False)), f"posting, round {rd}, {arm}"))
        worst.append((abs(q(m, arm, ("un", "cr", "y"), ("un", "cr", "n"), False)
                          - q(m, "none", ("un", "cr", "y"), ("un", "cr", "n"), False)), f"CR on unplanted questions, {arm}"))
    w = max(worst)
    say(f"- {lab}, no change where the plant cannot act: largest difference from none {w[0]:.1f} ({w[1]}) (within 5): "
        f"**{'met' if w[0] <= 5 else 'not met'}**.")
for lab, m in PM:
    r = res[m]
    say(f"- {lab}, the true arm, reached finders' use against none: {r['dt']:+.1f} (within 5): "
        f"**{'met' if abs(r['dt']) <= 5 else 'not met'}**.")
say()
say("**In short.** A wrong number on the page costs every model's finders use of the true value, by 16 to 39 points, "
    "and holders' posting and the unplanted questions do not move (the plant acts only where it was put). The cost "
    "is not fixed: it shrinks as more copies post the true value. With one copy's true value beside the planted guess, "
    "Qwen3-8B finders use it 33.5% of the time (94.5% with no plant); with three or more, 89.4% (100.0%). That is why "
    "the fuller model, which charges every guessed page the same cost, falls short at large N, where pages carry many "
    "true posts, and why the simple model, which ignores guesses, comes closer there. Qwen3-4B and 14B also use the "
    "value more when a planted note agrees with it (true arm +21.0 and +13.2), so for them agreement on the page "
    "matters as well as disagreement. Qwen3-4B departs most from its registered numbers: its cost of a guess "
    "(−15.9 against −1.7) and how often it takes the planted value (35.9 against 61.8).")
say()
say("Holders who post the true value, by round (all questions; a holder writes before it reads, so only a plant "
    "met in an earlier round can reach it):")
say()
say("| Model | Arm | Round 1 | 2 | 3 | 4 | CR, unplanted questions |")
say("|---|---|---|---|---|---|---|")
for lab, m in PM:
    for arm in ARMS:
        say(f"| {lab} | {arm} | " + " | ".join(f"{q(m, arm, ('post', rd, 'y'), ('post', rd, 'n'), False):.1f}" for rd in (1, 2, 3, 4))
            + f" | {q(m, arm, ('un', 'cr', 'y'), ('un', 'cr', 'n'), False):.1f} |")
(HERE / "results_pod5.md").write_text("\n".join(lines) + "\n")
print("wrote results_pod5.md")
