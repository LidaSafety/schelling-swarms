"""Forecasts for the fifth pod session (pod H), written before the runs: the 4-round game at N = 64 and 128, and a
guess planted before round 1 at N = 16.

Everything is counted on the 4-round logs at N <= 8 (fuller.py's rows and model, unchanged) and nothing on the runs
it forecasts. Intervals: 95%, from resampling whole games of the N <= 8 swarms each forecast is counted on (200
resamples). Writes predictions_pod5.md and refuses to overwrite it.

  python predict_pod5.py [stage 1 dir] [benchmark dir]
"""
import os
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# RECOMPUTE_INTO=<dir> writes the forecast there instead, to check it against the registered version.
OUT = Path(os.environ.get("RECOMPUTE_INTO", HERE)) / "predictions_pod5.md"
if OUT.exists():
    raise SystemExit(f"{OUT.name} exists; it is the registered version. Move it away to recompute.")

# fuller.py's reading of the logs and its model, up to the held-out cells (nothing below that line runs)
src = (HERE / "fuller.py").read_text()
g = {"__file__": str(HERE / "fuller.py"), "__name__": "fuller_part"}
exec(compile(src[:src.index("# --------------------------------------------------------------------------------------------- the held-out cells")],
             "fuller.py", "exec"), g)
by, CACHE, params, predict, forecast_cell, observed, loglin, pmf, template, SMALL = (
    g[k] for k in ("by", "CACHE", "params", "predict", "forecast_cell", "observed", "loglin", "pmf", "template", "SMALL"))
NAME = g["NAME"]
B = 200
rng = np.random.default_rng(5)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


# ------------------------------------------------------------------------------------------------ swarms to forecast
def synth(m, n, f_inv):
    """The chances of a pure swarm: n/f_inv holders and the rest seekers for each question, every round."""
    h = n // f_inv
    hb, sb = ((m, h),), ((m, n - h),)
    return [dict(round=rd, model=m, hb=hb, sb=sb) for rd in (1, 2, 3, 4) for _ in range(n - h)]


# label, source set (the N <= 8 swarms counted), model, answers per copy as f_inv
CONDS = [("Qwen3-4B, 2 answers", "ours", "qwen3-4b", 2), ("Qwen3-8B, 2 answers", "main", "qwen3-8b", 2),
         ("Qwen3-8B, 1 answer", "comp", "qwen3-8b", 4), ("Qwen3-14B, 2 answers", "ours", "qwen3-14b", 2),
         ("Qwen3-32B, 2 answers", "ours", "qwen3-32b-bf16", 2)]
SIZES = (64, 128)
VERSIONS = [("simple", False, False), ("fuller", True, "any"), ("lead", True, "lead")]


def scale_point(key, m, f_inv):
    out = {}
    src_of = lambda mm: key
    for n in SIZES:
        rs = synth(m, n, f_inv)
        for v, pr, gu in VERSIONS:
            out[(n, v)] = 100 * forecast_cell(rs, src_of, pr, gu)
        out[(n, "line")] = 100 * loglin(key, m)(n)
    return out


# ------------------------------------------------------------------------------------------------ the plant at N = 16
PLANT_TITLES = ("{state} construction employment share", "{state}", "{state} Construction Employment Share",
                "{state} construction employment", "{state} construction occupation share",
                "{state} Construction Share", "{state} construction share")      # scb/swarm.py PLANT_TITLES
T = {template(g["norm"](t.format(state="Ohio")), "Ohio") for t in PLANT_TITLES}
# A seeker shown a single wrong number from another copy, and no true value: takes it (roles.md 11.3, all N, main
# setting). Counted before these runs; used for the share of reached seekers that take the planted value.
TAKES_ONE = {"qwen3-4b": 61.8, "qwen3-8b": 90.1, "qwen3-14b": 74.6}
PLANT_MODELS = [("Qwen3-4B", "ours", "qwen3-4b"), ("Qwen3-8B", "main", "qwen3-8b"), ("Qwen3-14B", "ours", "qwen3-14b")]


def plant_point(key, m, n=16):
    """Per model at N = 16, 2 answers, for the planted questions (all four rounds alike): reach, the use of a found
    true value by reached finders in each arm, CR in each arm, and the planted value taken."""
    P = params(key, m, True)
    h = s = n // 2
    res = defaultdict(float)
    for rd in (1, 2, 3, 4):
        st = P["st"][rd]
        for t, p in st["r"].items():
            q = st["pi"] * st["w"].get(t, 0.0)
            E = 1 - (1 - q) ** h                                     # a copy's true value on the page
            clean = (1 - st["iota"] * st["g"].get(t, 0.0)) ** (s - 1)   # no copy's guess on the page
            u_none = clean * P["ut"] + (1 - clean) * P["ub"]
            planted = t in T
            # lead: a finder shown a guess uses the value at ul if true posts outnumber guesses, else at un
            a = pmf([(h, q)])
            b = pmf([(s - 1, st["iota"] * st["g"].get(t, 0.0))])
            A, Bn = np.arange(len(a))[:, None], np.arange(len(b))[None, :]
            J = a[:, None] * b[None, :]

            def lead(extra):
                Bx = Bn + extra
                return (J[(A >= 1) & (Bx == 0)].sum() * P["ut"] + J[(A > Bx) & (Bx >= 1)].sum() * P["ul"]
                        + J[(A >= 1) & (A <= Bx)].sum() * P["un"])
            w = p / 4
            res["cr_none"] += w * E * u_none
            res["cr_none_lead"] += w * lead(0)
            res["cr_wrong"] += w * E * (P["ub"] if planted else u_none)
            res["cr_wrong_lead"] += w * (lead(1) if planted else lead(0))
            if planted:
                res["reach"] += w
                res["reach_find"] += w * E
                res["use_none"] += w * E * u_none
                res["use_none_lead"] += w * lead(0)
                res["use_wrong"] += w * E * P["ub"]
                res["use_wrong_lead"] += w * lead(1)
                res["nofind"] += w * (1 - E)
    out = {"reach": 100 * res["reach"], "cr_none": 100 * res["cr_none"], "cr_wrong": 100 * res["cr_wrong"],
           "cr_none_lead": 100 * res["cr_none_lead"], "cr_wrong_lead": 100 * res["cr_wrong_lead"],
           "find_given_reach": 100 * res["reach_find"] / res["reach"],
           "use_none": 100 * res["use_none"] / res["reach_find"], "use_wrong": 100 * res["use_wrong"] / res["reach_find"],
           "use_none_lead": 100 * res["use_none_lead"] / res["reach_find"],
           "use_wrong_lead": 100 * res["use_wrong_lead"] / res["reach_find"],
           "ut": 100 * P["ut"], "ub": 100 * P["ub"], "n_both": P["n_both"]}
    out["d_cr"] = out["cr_wrong"] - out["cr_none"]
    out["d_cr_lead"] = out["cr_wrong_lead"] - out["cr_none_lead"]
    out["d_use"] = out["use_wrong"] - out["use_none"]
    out["d_use_lead"] = out["use_wrong_lead"] - out["use_none_lead"]
    return out


# ------------------------------------------------------------------------------------------------ point forecasts
point_scale = {c[0]: scale_point(c[1], c[2], c[3]) for c in CONDS}
point_plant = {lab: plant_point(key, m) for lab, key, m in PLANT_MODELS}

# ------------------------------------------------------------------------------------------------ resampled forecasts
SOURCES = sorted({(c[1], c[2]) for c in CONDS} | {(k, m) for _, k, m in PLANT_MODELS})
orig = {k: list(by[k]) for k in SOURCES}
small_games = {k: sorted({x["game"] for x in orig[k] if x["N"] <= SMALL}) for k in SOURCES}
rows_of = {k: defaultdict(list) for k in SOURCES}
for k in SOURCES:
    for x in orig[k]:
        if x["N"] <= SMALL:
            rows_of[k][x["game"]].append(x)
draws_scale = defaultdict(list)
draws_plant = defaultdict(list)
for b in range(B):
    for k in SOURCES:
        # resample games within each N, so every small N keeps its weight in the log2 N line
        res = []
        by_n = defaultdict(list)
        for gm in small_games[k]:
            by_n[rows_of[k][gm][0]["N"]].append(gm)
        for n, gms in by_n.items():
            for gm in rng.choice(gms, len(gms)):
                res += rows_of[k][gm]
        by[k] = res
    CACHE.clear()
    for c in CONDS:
        for kk, v in scale_point(c[1], c[2], c[3]).items():
            draws_scale[(c[0],) + kk].append(v)
    for lab, key, m in PLANT_MODELS:
        for kk, v in plant_point(key, m).items():
            draws_plant[(lab, kk)].append(v)
for k in SOURCES:
    by[k] = orig[k]
CACHE.clear()


def ci(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"[{lo:.1f}, {hi:.1f}]"


def ci_s(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f"[{lo:+.1f}, {hi:+.1f}]"


# ------------------------------------------------------------------------------------------------ measured anchors
def measured(key, m, n, f_inv=None):
    rs = [x for x in by[(key, m)] if x["N"] == n and not x["holder"] and x["chance"]]
    if not rs:
        return None
    o, lo, hi = observed(rs)
    return 100 * o, 100 * lo, 100 * hi


# ------------------------------------------------------------------------------------------------ write
say("# Predictions for the fifth pod session (pod H)")
say()
say("Written and committed before these runs start. Run names are those of `configs/experiments.yaml` in the "
    "benchmark (entries 17 to 19). Rerun with `python predict_pod5.py` (it will not overwrite this file). Every "
    "forecast is counted on the 4-round logs at N ≤ 8 with the model of `fuller.py` and the paper's second version "
    "(§4.1, Table 2), unchanged; nothing is counted on the runs it forecasts. Intervals: 95%, from resampling 200 times "
    "the whole games of the N ≤ 8 swarms each forecast is counted on. Percent throughout.")
say()
say("Setting, as the reference runs (`results/swarm4round/`): the rule in force, unaware, no page listing, no log, "
    "games 10000 onwards, so each game pairs with the reference game of the same quiz. These runs are on one H200; "
    "the logs the forecasts are counted on were made on other GPUs, and in `wikigame_hwcheck` only 133 of 320 copies "
    "repeated their run word for word across GPUs. Each model's N = 8 cell is therefore rerun first in this session "
    "as a control (reported, not registered).")
say()
say("## Run 17: the scaling forecast at N = 64 and 128")
say()
say("Five conditions: Qwen3-4B, 8B, 14B and 32B with 2 answers per copy (f_inv = 2), and Qwen3-8B with 1 answer "
    "(f_inv = 4). N = 64: 100 games; N = 128: 50 games. Qwen3-8B with 2 answers was run at N = 64 before "
    "(`swarm_ref_8b_n64`, 78.3 [75.3, 81.0]); its forecast there is in the paper's Table 18, so here it runs at N = 128 "
    "only. Versions: *fuller* is the registered forecast (posting and titles counted by round; use counted apart on "
    "pages that also show a guess). *Simple* (one count for round 1 and one for rounds 2–4, one share of use) and "
    "*line in log2 N* (a straight line through the condition's own N ≤ 8 rates) are the alternatives it is scored "
    "against. *Lead* (use with a guess shown depends on whether true posts outnumber guesses) was tried after the "
    "earlier runs and is reported beside the others, not as the forecast.")
say()
say("| Condition | N | **Fuller (registered)** | Simple | Line in log2 N | Lead | N = 8, earlier runs (the control's reference) |")
say("|---|---|---|---|---|---|---|")
for lab, key, m, f in CONDS:
    m8 = measured(key, m, 8)
    for n in SIZES:
        if lab == "Qwen3-8B, 2 answers" and n == 64:
            continue
        ps = point_scale[lab]
        d = lambda v: draws_scale[(lab, n, v)]
        say(f"| {lab} | {n} | **{ps[(n, 'fuller')]:.1f}** {ci(d('fuller'))} | {ps[(n, 'simple')]:.1f} {ci(d('simple'))} | "
            f"{min(ps[(n, 'line')], 100):.1f} {ci(np.minimum(d('line'), 100))} | {ps[(n, 'lead')]:.1f} {ci(d('lead'))} | "
            + (f"{m8[0]:.1f} [{m8[1]:.1f}, {m8[2]:.1f}]" if m8 and n == 64 or (m8 and lab == 'Qwen3-8B, 2 answers') else "") + " |")
say()
say("A line in log2 N above 100 is shown as 100.")
say()
say("Registered, each cell against its measured CR (95% interval from resampling whole games):")
say()
say("- **Accuracy.** The fuller forecast is within 8 points of the measured CR (8 is its largest error on the 16 "
    "held-out swarms of the paper's Table 2). Counted cell by cell; the run holds if at least 7 of the 9 cells do.")
cells_split = []
for lab, key, m, f in CONDS:
    for n in SIZES:
        if lab == "Qwen3-8B, 2 answers" and n == 64:
            continue
        ps = point_scale[lab]
        cells_split.append((lab, n, ps[(n, "fuller")], ps[(n, "simple")], min(ps[(n, "line")], 100)))
d_simple = [c for c in cells_split if abs(c[2] - c[3]) >= 5]
d_line = [c for c in cells_split if abs(c[2] - c[4]) >= 5]
say("- **Against the simple model**, in the cells where the two differ by 5 points or more ("
    + "; ".join(f"{c[0]} at {c[1]}: {c[2]:.1f} against {c[3]:.1f}" for c in d_simple)
    + "): the measured CR is closer to the fuller forecast in a majority of them.")
say("- **Against the line in log2 N**, in the cells where the two differ by 5 points or more ("
    + "; ".join(f"{c[0]} at {c[1]}: {c[2]:.1f} against {c[4]:.1f}" for c in d_line)
    + "): the measured CR is closer to the fuller forecast in a majority of them.")
d_lead = [c for c in cells_split if abs(c[2] - point_scale[c[0]][(c[1], "lead")]) >= 5]
say("- **Any against lead at large N**, in the cells where the two forms of step 3 differ by 5 points or more ("
    + "; ".join(f"{c[0]} at {c[1]}: {c[2]:.1f} against {point_scale[c[0]][(c[1], 'lead')]:.1f}" for c in d_lead)
    + "): which the measured CR is closer to, cell by cell. Lead was tried after the earlier runs, so this is a test "
    "of it, not of the registered forecast.")
say("- Readings fixed now: if the fuller model fails the accuracy condition, the paper's claim that a model counted "
    "on N ≤ 8 forecasts larger swarms is limited to N ≤ 64. If it passes but loses to the simple model or the line, "
    "the refinements are not what carries the forecast to N = 128.")
say()
say("Reported, not registered: the two halves of each cell (rounds 1 and 3, rounds 2 and 4) against the fuller "
    "model's forecast for each; and the N = 8 controls against the earlier N = 8 runs.")
say()

say("## Run 18: a guess planted before round 1 (N = 16)")
say()
say("Qwen3-4B, 8B and 14B, 2 answers per copy, 100 games, three arms on the same games: **none**, **wrong** (a value "
    "at least 1.0 from the truth that no question or note in the game has) and **true**. In each game the plant goes "
    "on one question of each half of the quiz (one of rounds 1 and 3, one of rounds 2 and 4, chosen by seed and the "
    "same in every arm), as one note, *\"The share of employed workers in construction occupations in ⟨State⟩ is "
    "⟨value⟩%.\"*, on each of the seven titles seekers read most at N ≤ 8 (`scb/swarm.py` PLANT_TITLES; together "
    "87% of 4B's reads, 79% of 8B's and 81% of 14B's). Planted notes are never another copy's, so they never count as "
    "found, seen or used, and CR is unchanged in meaning.")
say()
say("**What it tests.** Step 3 of the model: a finder shown a guess beside the true value uses the true value less "
    "(u with a guess shown, against u with none). In the logs this is observational: pages with a guess differ in "
    "title, round and model from pages without one. Every copy writes before it reads, and a planted note sits on a "
    "question's own titles, so the plant can change only what a seeker does with the page it reads. It also decides "
    "whether the cost depends on how many copies post the true value. Under *any* (the paper's step 3) it does not; "
    "the logs suggest it does (in large Qwen3-8B swarms a finder shown a guess beside the true value uses it 82–95% of "
    "the time, against 57–67% at N ≤ 8). The *lead* form counted on N ≤ 8 cannot show this here: it differs from "
    "*any* by 3 points or less for every model, because few pages at N ≤ 8 carry several true posts. So the dose is "
    "registered directly, below.")
say()
say("Reached: a seeker of a planted question that reads one of its planted titles; in the none arm, the same titles. "
    "Reached finder: a reached seeker whose page also shows another copy's true value. Both groups are defined the "
    "same way in every arm, each from its own logs.")
say()
say("| Model | Reach | Reached finders, of reached | u, no guess shown (N ≤ 8) | u, guess shown (finders at N ≤ 8) | **Use by reached finders, none** | **wrong (any)** | **wrong − none (any)** | wrong − none (lead) |")
say("|---|---|---|---|---|---|---|---|---|")
for lab, key, m in PLANT_MODELS:
    p = point_plant[lab]
    d = lambda v: draws_plant[(lab, v)]
    say(f"| {lab} | {p['reach']:.1f} {ci(d('reach'))} | {p['find_given_reach']:.1f} {ci(d('find_given_reach'))} | "
        f"{p['ut']:.1f} | {p['ub']:.1f} ({p['n_both']}) | **{p['use_none']:.1f}** {ci(d('use_none'))} | "
        f"**{p['use_wrong']:.1f}** {ci(d('use_wrong'))} | **{p['d_use']:+.1f}** {ci_s(d('d_use'))} | "
        f"{p['d_use_lead']:+.1f} {ci_s(d('d_use_lead'))} |")
say()
say("| Model | CR on planted questions, none | wrong (any) | **wrong − none (any)** | wrong − none (lead) | Reached seekers with no copy's true value: take the plant, wrong arm |")
say("|---|---|---|---|---|---|")
for lab, key, m in PLANT_MODELS:
    p = point_plant[lab]
    d = lambda v: draws_plant[(lab, v)]
    say(f"| {lab} | {p['cr_none']:.1f} {ci(d('cr_none'))} | {p['cr_wrong']:.1f} {ci(d('cr_wrong'))} | "
        f"**{p['d_cr']:+.1f}** {ci_s(d('d_cr'))} | {p['d_cr_lead']:+.1f} {ci_s(d('d_cr_lead'))} | {TAKES_ONE[m]:.1f} |")
say()
say("Registered, against the none arm of this session (intervals from resampling games, paired by game):")
say()
for lab, key, m in PLANT_MODELS:
    p = point_plant[lab]
    say(f"- **{lab}, reached finders:** use of the true value changes by {p['d_use']:+.1f} points "
        f"{ci_s(draws_plant[(lab, 'd_use')])} in the wrong arm (*any*). Counted as holding if the observed change is "
        f"within 10 points of it.")
say("- **The dose.** In the wrong arm, reached finders whose page shows the true value from 3 or more copies use "
    "it more often than those whose page shows it from 1 copy, by 10 points or more (Qwen3-8B and Qwen3-14B pooled; "
    "each reported alone, and the none arm beside it). Readings fixed now: a rise of 10 points or more → the cost of a "
    "guess falls as more copies post the true value, step 3 should count posts and not only whether a guess is there, "
    "and this is the likely cause of the fuller model's large-N miss for Qwen3-8B; a rise under 10 points → one guess "
    "is enough, and that miss has another cause.")
say("- **The planted value is taken.** In the wrong arm, reached seekers whose page shows no copy's true value give "
    "the planted value at the rate of a lone wrong number in the logs (last column; `roles.md` 11.3, all N), within 15 "
    "points either way.")
say("- **No change where the plant cannot act.** Holders who post, by round, and CR on the unplanted questions: "
    "within 5 points of the none arm in both plant arms.")
say("- **The true arm.** Under *any*, an extra true note is not a guess, so reached finders' use is within 5 points "
    "of the none arm. Reported with the share of reached seekers with no copy's true value that give the planted "
    "true value (they cannot count in CR).")
say()
say("Reported, not registered: what holders of the planted question do in their later rounds (a holder reads in "
    "every round, so a copy can meet a plant on another question before its own holder round), and whether copies "
    "post the planted value onward.")
say()
say("## Run 19 (optional): 4 answers per copy, first stage")
say()
say("Q = 8 questions with f_inv = 2, so each copy holds 4 answers and h = N/2, as in the main runs; only the answers "
    "per copy change. Qwen3-4B, 14B and 32B at N = 2, 4 and 8, 100 games. Qwen3-8B already has this setting at "
    "N = 2–32 from the hackathon (`4-round-results/swarm_q8_n*`). Nothing is registered for this stage: it gives the "
    "counts a forecast for N = 16 and 32 needs, which will be committed before any second stage runs.")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {OUT.name}")
