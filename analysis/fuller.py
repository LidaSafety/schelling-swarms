"""Does the fuller model of the 4-round game predict large swarms better than the simple one?

The hackathon paper's second version models a use as three steps: a holder posts (step 1), a seeker finds the page
(step 2), and the seeker uses the value (step 3). Its Section 4 counts each step once per model on small swarms. Its
Section 5 finds three ways the counts depend on more than the model: posting depends on the role a copy had first,
use depends on whether the page also shows a guess, and the titles change after round 1. This script puts those
refinements back into the forecast and scores every version on the swarms held out from the count (N > 8).

Every forecast is counted on swarms of N <= 8 of the same set and model and applied to the larger swarms. Nothing is
fitted to the held-out swarms.

  N alone         CR = a + b log2 N, fitted to the collusion rates at N <= 8 and extrapolated. Also one curve for
                  every condition: the Qwen3-8B main runs' fit.
  Simple          CR = E x u. E = 1 - sum_t r_t prod_m (1 - pi_m w^m_t)^h_m (the paper's equation 1, with posting
                  forecast). r, w and pi are counted for round 1 and for rounds 2-4 pooled; u is one share per model,
                  "found -> used", over all rounds.
  + first role    r, w and pi counted for each round on its own, so a round whose holders began as seekers gets its
                  own posting rate and its own titles.
  + guesses       step 3 depends on the page. Each other seeker posts a wrong number with chance iota on title t with
                  chance g_t, so the page a seeker reads also shows a guess with chance
                  P_t = 1 - prod_m (1 - iota_m g^m_t)^(s_m - [m is the reader's model]); a finder uses the value with
                  u_true on a page with no guess and u_both on a page with one. CR = sum_t r_t E_t ((1-P_t) u_true + P_t u_both).
  Fuller          both refinements.
  Fuller, lead    tried after seeing the held-out swarms, and reported only as a check: a finder shown a guess uses
                  the value with one share if the true value is posted more often than guesses on its page, and
                  another if not. The numbers of true posts and guesses are binomial with the chances above.

  python fuller.py [stage 1 dir] [benchmark dir]      (writes fuller.md)

Counted after the runs. The definitions of a post, a wrong number and what a page shows are those of roles.py.
"""
import json
import math
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES  # noqa: E402
BENCH = ROOT
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))   # model client only; not needed here
sys.path.insert(0, str(BENCH))
from scb.io import read_jsonl  # noqa: E402
from scb.wikigame import _NUM, state_named  # noqa: E402

LO, HI = 4.0, 12.0
SMALL = 8
OURS = RES / "swarm4round"
SETS = {
    "main": sorted(STAGE1.glob("swarm_main_n*")),              # Qwen3-8B, 2 answers per copy (Figure 1)
    "comp": sorted(STAGE1.glob("swarm_comp_n*")),              # Qwen3-8B, 1 answer per copy (Figure 2)
    "gemma": sorted(STAGE1.glob("swarm_gemma_n*")),            # Gemma 3 27B alone (Figure 3)
    "mix": sorted(STAGE1.glob("swarm_mix_n*")),                # Qwen3-8B with Gemma 3 27B (Figure 3)
    "ours": sorted(OURS.glob("swarm_ref*")),                   # 4B, 14B, 32B (Figure 4); the 8B rerun and N = 64
    "norule": [RES / "swarm_ruleoff", RES / "swarm_ruleoff2"],  # the same quizzes without the rule (Section 4.5)
}
NAME = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B", "qwen3-32b-bf16": "Qwen3-32B",
        "gemma-3-27b": "Gemma 3 27B"}
rng = np.random.default_rng(7)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def norm(t):
    t = " ".join(str(t or "").replace("_", " ").split())
    return t[:1].upper() + t[1:]


def template(title, state):
    return None if title is None else re.sub(re.escape(state), "<S>", norm(title), flags=re.I)


def nums(text):
    return [round(float(x), 1) for x in _NUM.findall(str(text or "")) if LO <= float(x) <= HI]


def writes(ag):
    """The note a copy wrote in each round, (title, text) or None, read from its transcript (as in roles.py)."""
    out, rd, step = {}, None, None
    for m in ag["messages"][1:]:
        c = m["content"]
        if m["role"] == "user":
            mm = re.search(r"ROUND (\d+)/", c)
            if mm:
                rd = int(mm.group(1))
            step = "write" if "WRITE step" in c else ("read" if "READ step" in c else ("answer" if "ANSWER step" in c else step))
            continue
        if m["role"] != "assistant" or step != "write" or rd is None:
            continue
        mm = re.search(r"\{.*\}", c, re.S)
        try:
            obj = json.loads(mm.group()) if mm else None
        except json.JSONDecodeError:
            obj = None
        ed = obj.get("edit") if isinstance(obj, dict) else None
        out[rd] = (norm(ed["title"]), str(ed.get("text") or "")) if isinstance(ed, dict) and ed.get("title") else None
        step = None
    return out


# ------------------------------------------------------------------------------------------------------------ read
rows = []                    # one per copy per round
gid = 0
for key, dirs in SETS.items():
    for d in dirs:
        for f in sorted((d / "raw").glob("*.jsonl*")):
            for r in read_jsonl(f):
                if ("error" in r or r.get("rule") != ("silent" if key == "norule" else "forbidden")
                        or r.get("awareness") != "unaware"):
                    continue
                gid += 1
                n, label = r["n_agents"], r["model"]
                seat = r.get("seat_models") or {}
                model = {a: (seat.get(a) if isinstance(seat, dict) else None) or r["model"] for a in r["agents"]}
                if isinstance(seat, list):
                    model = dict(zip(sorted(r["agents"]), seat))
                wrote = {a: writes(ag) for a, ag in r["agents"].items()}
                read = {(a, x["round"]): norm(x["path"]) if x.get("path") else None
                        for a, ev in r["events"].items() for x in ev["reads"]}
                hold = {a: set(v) for a, v in r["holds"].items()}
                for a, ag in r["agents"].items():
                    for ans in ag["answers"]:
                        rd, st, truth = ans["round"], ans["state"], ans["truth"]
                        w = wrote[a].get(rd)
                        kind = "skip"
                        if w is not None:
                            if state_named(st, w[0], w[1]):
                                wn = nums(w[1])
                                kind = "true" if any(abs(x - truth) < 1e-6 for x in wn) else "wrong" if wn else "words"
                            else:
                                kind = "other page"
                        page = read.get((a, rd))
                        n_true, n_wrong = 0, 0
                        if page:
                            for b in r["agents"]:
                                if b == a:
                                    continue
                                for w_rd, wb in wrote[b].items():
                                    if wb is None or w_rd > rd or wb[0] != page or not state_named(st, wb[0], wb[1]):
                                        continue
                                    ns = nums(wb[1])
                                    if any(abs(x - truth) < 1e-6 for x in ns):
                                        n_true += 1
                                    elif ns:
                                        n_wrong += 1
                        hb = Counter(model[b] for b in hold if st in hold[b])
                        sb = Counter(model[b] for b in hold if st not in hold[b])
                        rows.append(dict(set=key, mix=label, model=model[a], N=n, game=gid, round=rd,
                                         holder=st in hold[a], kind=kind, w_tpl=template(w[0], st) if w else None,
                                         r_tpl=template(page, st), true_shown=n_true > 0, wrong_shown=n_wrong > 0, n_true=n_true, n_wrong=n_wrong,
                                         used=bool(ans["used_other"]), chance=bool(ans["chance"]),
                                         hb=tuple(sorted(hb.items())), sb=tuple(sorted(sb.items()))))
print(f"{len(rows):,} copy-rounds from {gid:,} games")

# Index rows by (set, model) for speed.
by = defaultdict(list)
for x in rows:
    by[(x["set"], x["model"])].append(x)


# ------------------------------------------------------------------------------------------------- the counts
def stage_of(rd, per_round):
    return rd if per_round else min(rd, 2)


def counts(key, m, per_round):
    """Everything each forecast needs, counted on the small swarms of one set and model."""
    src = [x for x in by[(key, m)] if x["N"] <= SMALL and (key != "ours" or True)]
    out = defaultdict(dict)
    stages = sorted({stage_of(x["round"], per_round) for x in src})
    for s in stages:
        S = [x for x in src if stage_of(x["round"], per_round) == s]
        seek = [x for x in S if not x["holder"] and x["chance"]]
        hold = [x for x in S if x["holder"]]
        posters = [x for x in hold if x["kind"] == "true"]
        guessers = [x for x in seek if x["kind"] == "wrong"]
        rc = Counter(x["r_tpl"] for x in seek)
        out[s]["r"] = {t: v / len(seek) for t, v in rc.items()} if seek else {}
        wc = Counter(x["w_tpl"] for x in posters)
        out[s]["w"] = {t: v / len(posters) for t, v in wc.items()} if posters else {}
        out[s]["pi"] = len(posters) / len(hold) if hold else 0.0
        gc = Counter(x["w_tpl"] for x in guessers)
        out[s]["g"] = {t: v / len(guessers) for t, v in gc.items()} if guessers else {}
        out[s]["iota"] = len(guessers) / len(seek) if seek else 0.0
    seek = [x for x in src if not x["holder"] and x["chance"]]
    found = [x for x in seek if x["true_shown"]]
    clean = [x for x in found if not x["wrong_shown"]]
    both = [x for x in found if x["wrong_shown"]]
    u = sum(x["used"] for x in found) / len(found) if found else 0.0
    ut = sum(x["used"] for x in clean) / len(clean) if clean else u
    ub = sum(x["used"] for x in both) / len(both) if len(both) >= 20 else ut
    lead = [x for x in both if x["n_true"] > x["n_wrong"]]
    trail = [x for x in both if x["n_true"] <= x["n_wrong"]]
    ul = sum(x["used"] for x in lead) / len(lead) if len(lead) >= 20 else ut
    un = sum(x["used"] for x in trail) / len(trail) if len(trail) >= 20 else ub
    return dict(st=out, u=u, ut=ut, ub=ub, ul=ul, un=un, n_both=len(both), n_lead=len(lead), n_trail=len(trail),
                n_found=len(found))


def pmf(parts):
    """The distribution of a sum of independent binomials, parts = [(trials, chance)]."""
    out = np.array([1.0])
    for k, q in parts:
        k = int(k)
        if k <= 0 or q <= 0:
            continue
        b = np.array([math.comb(k, i) * q ** i * (1 - q) ** (k - i) for i in range(k + 1)])
        out = np.convolve(out, b)
    return out


SRC = {"qwen3-8b": "main", "gemma-3-27b": "gemma", "qwen3-4b": "ours", "qwen3-14b": "ours", "qwen3-32b-bf16": "ours"}
CACHE = {}


def params(key, m, per_round):
    k = (key, m, per_round)
    if k not in CACHE:
        CACHE[k] = counts(key, m, per_round)
    return CACHE[k]


def predict(x, src_of, per_round, guesses):
    """Forecast use for one chance: x gives its set, round, reader model and the holders and seekers by model."""
    me = x["model"]
    P = {m: params(src_of(m), m, per_round) for m, _ in x["hb"] + x["sb"]}
    P.setdefault(me, params(src_of(me), me, per_round))
    s = stage_of(x["round"], per_round)
    r = P[me]["st"][s]["r"]
    total = 0.0
    for t, p in r.items():
        miss = 1.0
        for m, h in x["hb"]:
            miss *= (1 - P[m]["st"][s]["pi"] * P[m]["st"][s]["w"].get(t, 0.0)) ** h
        E = 1 - miss
        if not guesses:
            total += p * E * P[me]["u"]
            continue
        if guesses == "any":
            clean = 1.0
            for m, k in x["sb"]:
                k = k - (1 if m == me else 0)
                clean *= (1 - P[m]["st"][s]["iota"] * P[m]["st"][s]["g"].get(t, 0.0)) ** k
            total += p * E * (clean * P[me]["ut"] + (1 - clean) * P[me]["ub"])
            continue
        a = pmf([(h, P[m]["st"][s]["pi"] * P[m]["st"][s]["w"].get(t, 0.0)) for m, h in x["hb"]])
        b = pmf([(k - (1 if m == me else 0), P[m]["st"][s]["iota"] * P[m]["st"][s]["g"].get(t, 0.0))
                 for m, k in x["sb"]])
        A, B = np.arange(len(a))[:, None], np.arange(len(b))[None, :]
        J = a[:, None] * b[None, :]
        total += p * (J[(A >= 1) & (B == 0)].sum() * P[me]["ut"] + J[(A > B) & (B >= 1)].sum() * P[me]["ul"]
                      + J[(A >= 1) & (A <= B)].sum() * P[me]["un"])
    return total


MODELS = [("Simple", False, False), ("+ first role", True, False), ("+ guesses", False, "any"), ("Fuller", True, "any"),
          ("Fuller, lead", True, "lead")]


def observed(rs):
    per = defaultdict(lambda: [0, 0])
    for x in rs:
        per[x["game"]][0] += x["used"]
        per[x["game"]][1] += 1
    a = np.array(list(per.values()), float)
    draws = a[rng.integers(0, len(a), (2000, len(a)))].sum(1)
    lo, hi = np.percentile(draws[:, 0] / draws[:, 1], [2.5, 97.5])
    return a[:, 0].sum() / a[:, 1].sum(), lo, hi


def forecast_cell(rs, src_of, per_round, guesses):
    memo, tot = {}, 0.0
    for x in rs:
        k = (x["round"], x["model"], x["hb"], x["sb"])
        if k not in memo:
            memo[k] = predict(x, src_of, per_round, guesses)
        tot += memo[k]
    return tot / len(rs)


def loglin(key, m):
    """CR = a + b log2 N through the small swarms' collusion rates."""
    xs, ys = [], []
    for n in sorted({x["N"] for x in by[(key, m)] if x["N"] <= SMALL}):
        rs = [x for x in by[(key, m)] if x["N"] == n and not x["holder"] and x["chance"]]
        xs.append(np.log2(n))
        ys.append(np.mean([x["used"] for x in rs]))
    b, a = np.polyfit(xs, ys, 1)
    return lambda n: a + b * np.log2(n)


# --------------------------------------------------------------------------------------------- the held-out cells
# (label, set the cell comes from, model, set the small-swarm counts come from, N)
CELLS = [("8B, 2 answers", "main", "qwen3-8b", "main", 16), ("8B, 2 answers", "main", "qwen3-8b", "main", 32),
         ("8B, 2 answers, rerun", "ours", "qwen3-8b", "main", 16), ("8B, 2 answers, rerun", "ours", "qwen3-8b", "main", 64),
         ("8B, 1 answer", "comp", "qwen3-8b", "comp", 16), ("8B, 1 answer", "comp", "qwen3-8b", "comp", 32)]
CELLS += [(f"{NAME[m][4:]}", "ours", m, "ours", n) for m in ("qwen3-4b", "qwen3-14b", "qwen3-32b-bf16") for n in (16, 32)]
HALVES = [("all rounds", (1, 2, 3, 4)), ("rounds 1 and 3", (1, 3)), ("rounds 2 and 4", (2, 4))]

one_curve = loglin("main", "qwen3-8b")
res = []
for lab, key, m, src, n in CELLS:
    for hl, rds in HALVES:
        rs = [x for x in by[(key, m)] if x["N"] == n and not x["holder"] and x["chance"] and x["round"] in rds]
        o, lo, hi = observed(rs)
        row = dict(lab=lab, N=n, half=hl, obs=o, lo=lo, hi=hi)
        if hl == "all rounds":
            row["N alone"] = loglin(src, m)(n)
            row["One curve"] = one_curve(n)
        for name, pr, gu in MODELS:
            row[name] = forecast_cell(rs, lambda mm, s=src: s if mm == m else SRC[mm], pr, gu)
        res.append(row)

say("# The fuller model against the simple one, on swarms held out from the count\n")
say("Every forecast is counted on swarms of N ≤ 8 of the same set and model and applied to the larger swarms; "
    "nothing is fitted to them. Collusion rate in percent. Observed: 95% interval from resampling whole games. "
    "*N alone*: a straight line in log2 N through the condition's own small swarms. *One curve*: the same line for "
    "the Qwen3-8B main runs, used for every condition. *Simple*: the paper's equation 1 with posting forecast, "
    "titles and posting counted for round 1 and for rounds 2–4, times one share of finders who use the value. "
    "*+ first role*: titles and posting counted for each round on its own. *+ guesses*: use depends on whether "
    "the page also shows a seeker's guess. *Fuller*: both. *Fuller, lead*: tried after seeing the held-out swarms, "
    "as a check; use with a guess shown depends on whether the true value is posted more often than guesses. "
    "Rerun with `python fuller.py`.\n")

say("## 1. The parameters, counted on N ≤ 8\n")
say("| Model | Set | π by round (1, 2, 3, 4) | ι by round | u, all finders | u, no guess shown | u, guess shown (finders) |")
say("|---|---|---|---|---|---|---|")
for key, m in (("main", "qwen3-8b"), ("comp", "qwen3-8b"), ("ours", "qwen3-4b"), ("ours", "qwen3-14b"),
               ("ours", "qwen3-32b-bf16"), ("gemma", "gemma-3-27b")):
    P = params(key, m, True)
    pis = ", ".join(f"{100 * P['st'][s]['pi']:.0f}" for s in sorted(P["st"]))
    ios = ", ".join(f"{100 * P['st'][s]['iota']:.0f}" for s in sorted(P["st"]))
    say(f"| {NAME[m]} | {key} | {pis} | {ios} | {100 * P['u']:.1f} | {100 * P['ut']:.1f} | "
        f"{100 * P['ub']:.1f} ({P['n_both']}) |")
say("\nWith fewer than 20 finders shown a guess, u with a guess shown is set to u with none.\n")

COLS = ["N alone", "One curve"] + [n for n, _, _ in MODELS]
say("## 2. Each held-out cell, all rounds\n")
say("| Condition | N | Observed | " + " | ".join(COLS) + " |")
say("|---|---|---|" + "---|" * len(COLS))
for x in res:
    if x["half"] != "all rounds":
        continue
    say(f"| {x['lab']} | {x['N']} | {100 * x['obs']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}] | "
        + " | ".join(f"{100 * x[c]:.1f}" for c in COLS) + " |")

say("\n## 3. The two halves of each cell\n")
say("| Condition | N | Half | Observed | " + " | ".join(n for n, _, _ in MODELS) + " |")
say("|---|---|---|---|" + "---|" * len(MODELS))
for x in res:
    if x["half"] == "all rounds":
        continue
    say(f"| {x['lab']} | {x['N']} | {x['half']} | {100 * x['obs']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}] | "
        + " | ".join(f"{100 * x[n]:.1f}" for n, _, _ in MODELS) + " |")


def score(sel, cols):
    out = []
    for c in cols:
        g = [x for x in res if sel(x) and c in x]
        e = np.array([100 * (x[c] - x["obs"]) for x in g])
        inside = sum(x["lo"] <= x[c] <= x["hi"] for x in g)
        out.append((c, len(g), np.median(np.abs(e)), np.abs(e).max(), e.mean(), inside))
    return out


say("\n## 4. Errors by group\n")
say("Error: forecast minus observed, in points.\n")
GROUPS = [("Qwen3-8B, all rounds (6 cells)", lambda x: x["lab"].startswith("8B") and x["half"] == "all rounds"),
          ("4B, 14B, 32B, all rounds (6 cells)", lambda x: not x["lab"].startswith("8B") and x["half"] == "all rounds"),
          ("Every cell, all rounds (12)", lambda x: x["half"] == "all rounds"),
          ("Every cell, by half (24)", lambda x: x["half"] != "all rounds")]
say("| Cells | Forecast | Median error | Largest error | Mean signed error | Inside the observed interval |")
say("|---|---|---|---|---|---|")
for lab, sel in GROUPS:
    cols = COLS if "by half" not in lab else [n for n, _, _ in MODELS]
    for c, k, med, mx, mean, ins in score(sel, cols):
        say(f"| {lab} | {c} | {med:.1f} | {mx:.1f} | {mean:+.1f} | {ins} of {k} |")

# ------------------------------------------------------------------------------------------------- mixed swarms
say("\n## 5. Mixed swarms, from the pure swarms\n")
say("Each model's counts come from its own pure swarms at N ≤ 8 (Qwen3-8B from the main runs, Gemma 3 27B from its "
    "own). *Pure swarms' average*: the paper's dashed line, the pure swarms' rates at the same N weighted by model "
    "share.\n")
pure_cr = {}
for key, m in (("main", "qwen3-8b"), ("gemma", "gemma-3-27b")):
    for n in (4, 8):
        rs = [x for x in by[(key, m)] if x["N"] == n and not x["holder"] and x["chance"]]
        pure_cr[(m, n)] = np.mean([x["used"] for x in rs])
mres = []
for lab in sorted({x["mix"] for x in rows if x["set"] == "mix"}):
    parts = {k: int(v) for k, v in (p.split(":") for p in lab.split(","))}
    n = sum(parts.values())
    rs = [x for x in rows if x["set"] == "mix" and x["mix"] == lab and not x["holder"] and x["chance"]]
    o, lo, hi = observed(rs)
    row = dict(lab=" + ".join(f"{v} {NAME[k]}" for k, v in parts.items()), N=n, obs=o, lo=lo, hi=hi,
               avg=sum(parts[k] / n * pure_cr[(k, n)] for k in parts), one=one_curve(n))
    for name, pr, gu in MODELS:
        row[name] = forecast_cell(rs, lambda mm: SRC[mm], pr, gu)
    mres.append(row)
say("| Swarm | Observed | One curve | Pure swarms' average | " + " | ".join(n for n, _, _ in MODELS) + " |")
say("|---|---|---|---|" + "---|" * len(MODELS))
for x in mres:
    say(f"| {x['lab']} | {100 * x['obs']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}] | {100 * x['one']:.1f} | {100 * x['avg']:.1f} | "
        + " | ".join(f"{100 * x[n]:.1f}" for n, _, _ in MODELS) + " |")
say("\n| Forecast | Median error | Largest error | Inside |")
say("|---|---|---|---|")
for c in ["one", "avg"] + [n for n, _, _ in MODELS]:
    e = np.array([100 * (x[c] - x["obs"]) for x in mres])
    say(f"| {dict(one='One curve', avg='Pure swarms average').get(c, c)} | {np.median(np.abs(e)):.1f} | {np.abs(e).max():.1f} | "
        f"{sum(x['lo'] <= x[c] <= x['hi'] for x in mres)} of {len(mres)} |")

# ------------------------------------------------------------------------------------------ swarms not yet run
say("\n## 6. Forecasts for swarms not yet run (draft; register before any run)\n")
say("The same counts on N ≤ 8, applied to swarms that have not been run. Each block of N/2 copies holds one half of "
    "the answers; in a mix each block has the swarm's share of each model. Collusion rate in percent, over all "
    "seekers, then the two halves.\n")


def synth(parts, n):
    """Chances of a swarm of n copies with model counts parts, 2 answers per copy: one row per round and reader model."""
    half = {m: k // 2 for m, k in parts.items()}
    hb = tuple(sorted((m, k) for m, k in half.items() if k))
    out = []
    for rd in (1, 2, 3, 4):
        for m, k in half.items():
            out += [dict(round=rd, model=m, hb=hb, sb=hb)] * k
    return out


def fc(parts, n, pr, gu, rds=(1, 2, 3, 4)):
    rs = [x for x in synth(parts, n) if x["round"] in rds]
    return forecast_cell(rs, lambda mm: SRC[mm], pr, gu)


say("| Swarm | N | Simple | + first role | + guesses | Fuller | Fuller, rounds 1 and 3 | Fuller, rounds 2 and 4 |")
say("|---|---|---|---|---|---|---|---|")
for m in ("qwen3-8b", "qwen3-4b", "qwen3-14b", "qwen3-32b-bf16", "gemma-3-27b"):
    for n in ((64, 128, 256) if m == "qwen3-8b" else (16, 32) if m == "gemma-3-27b" else (64, 128)):
        say(f"| {NAME[m]} | {n} | {100 * fc({m: n}, n, False, False):.1f} | {100 * fc({m: n}, n, True, False):.1f} | "
            f"{100 * fc({m: n}, n, False, 'any'):.1f} | {100 * fc({m: n}, n, True, 'any'):.1f} | "
            f"{100 * fc({m: n}, n, True, 'any', (1, 3)):.1f} | {100 * fc({m: n}, n, True, 'any', (2, 4)):.1f} |")
say("\nMixed swarms of Qwen3-8B and Gemma 3 27B at N = 16 and 32, with the weighted average of the forecast pure "
    "swarms (the paper's dashed line) and the sag, forecast minus average:\n")
say("| Swarm | N | Fuller | Weighted average of the pure forecasts | Sag |")
say("|---|---|---|---|---|")
for n in (8, 16, 32):
    pq, pg = fc({"qwen3-8b": n}, n, True, "any"), fc({"gemma-3-27b": n}, n, True, "any")
    for g in (n // 4, n // 2, 3 * n // 4):
        f = fc({"qwen3-8b": n - g, "gemma-3-27b": g}, n, True, "any")
        avg = (n - g) / n * pq + g / n * pg
        say(f"| {n - g} Qwen3-8B + {g} Gemma 3 27B | {n} | {100 * f:.1f} | {100 * avg:.1f} | {100 * (f - avg):+.1f} |")

# ------------------------------------------------------------------------------------------ the rule, by round
say("\n## 7. The rule, by round (N = 16)\n")
say("The paper's Section 4.5 removed the rule sentence and kept everything else, paired by quiz. The registered "
    "expectation was that 4B and 8B would move by less than 5 points. 8B rose by 17. Here each step is split by "
    "round: in rounds 1 and 3 the holders began as holders, in rounds 2 and 4 as seekers. Title match: Σ_t r_t w_t "
    "over the seekers and posting holders of those rounds. Percent, except the title match.\n")
say("| Model | Rule | Rounds | Holders who post | Seekers who post a guess | Title match | Found | Found → used | CR |")
say("|---|---|---|---|---|---|---|---|---|")
for m in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b-bf16"):
    for key, lab in (("ours", "yes"), ("norule", "no")):
        R = [x for x in by[(key, m)] if x["N"] == 16]
        for rds, rl in (((1, 3), "1 and 3"), ((2, 4), "2 and 4")):
            S = [x for x in R if x["round"] in rds and not x["holder"] and x["chance"]]
            H = [x for x in R if x["round"] in rds and x["holder"]]
            P = [x for x in H if x["kind"] == "true"]
            rc, wc = Counter(x["r_tpl"] for x in S), Counter(x["w_tpl"] for x in P)
            match = sum(rc[t] / len(S) * wc[t] / len(P) for t in rc) if P else 0.0
            F = [x for x in S if x["true_shown"]]
            say(f"| {NAME[m]} | {lab} | {rl} | {100 * len(P) / len(H):.1f} | "
                f"{100 * np.mean([x['kind'] == 'wrong' for x in S]):.1f} | {match:.2f} | "
                f"{100 * len(F) / len(S):.1f} | {100 * np.mean([x['used'] for x in F]) if F else float('nan'):.1f} | "
                f"{100 * np.mean([x['used'] for x in S]):.1f} |")

# ------------------------------------------------------------------------------------- curves for the figure
def synth1(m, n):
    """Chances of a pure swarm of n copies with 1 answer per copy: n/4 holders and 3n/4 seekers of each answer."""
    hb, sb = ((m, n // 4),), ((m, 3 * n // 4),)
    return [dict(round=rd, model=m, hb=hb, sb=sb) for rd in (1, 2, 3, 4) for _ in range(3 * n // 4)]


GRID = (2, 4, 8, 16, 32, 64, 128)
PANELS = [("8B, 2 answers", "main", "qwen3-8b", 2), ("8B, 1 answer", "comp", "qwen3-8b", 1),
          ("4B", "ours", "qwen3-4b", 2), ("14B", "ours", "qwen3-14b", 2), ("32B", "ours", "qwen3-32b-bf16", 2)]
curves = {}
for lab, key, m, ans in PANELS:
    src_of = lambda mm, s=key: s if mm == m else SRC[mm]
    obs = []
    for k2 in ((key, "ours") if lab == "8B, 2 answers" else (key,)):
        for n in sorted({x["N"] for x in by[(k2, m)]}):
            rs = [x for x in by[(k2, m)] if x["N"] == n and not x["holder"] and x["chance"]]
            o, lo, hi = observed(rs)
            obs.append(dict(N=n, set=k2, obs=100 * o, lo=100 * lo, hi=100 * hi))
    grid = [n for n in GRID if n >= (4 if ans == 1 else 2)]
    mk = (lambda n: synth1(m, n)) if ans == 1 else (lambda n: synth({m: n}, n))
    curves[lab] = dict(observed=obs, N=grid,
                       own_line=[100 * loglin(key, m)(n) for n in grid],
                       line_8b=[100 * one_curve(n) for n in grid],
                       simple=[100 * forecast_cell(mk(n), src_of, False, False) for n in grid],
                       fuller=[100 * forecast_cell(mk(n), src_of, True, "any") for n in grid])
mixpts = []
for g in (0, 2, 4, 6, 8):
    parts = {k: v for k, v in (("qwen3-8b", 8 - g), ("gemma-3-27b", g)) if v}
    if g in (0, 8):
        key, m = ("main", "qwen3-8b") if g == 0 else ("gemma", "gemma-3-27b")
        rs = [x for x in by[(key, m)] if x["N"] == 8 and not x["holder"] and x["chance"]]
        o, lo, hi = observed(rs)
    else:
        x0 = next(x for x in mres if x["N"] == 8 and x["lab"].startswith(f"{8 - g} Qwen3-8B"))
        o, lo, hi = x0["obs"], x0["lo"], x0["hi"]
    mixpts.append(dict(gemma=g, obs=100 * o, lo=100 * lo, hi=100 * hi,
                       avg=100 * ((8 - g) / 8 * pure_cr[("qwen3-8b", 8)] + g / 8 * pure_cr[("gemma-3-27b", 8)]),
                       simple=100 * fc(parts, 8, False, False), fuller=100 * fc(parts, 8, True, "any")))
curves["mix, N = 8"] = mixpts
(HERE / "out" / "curves.json").write_text(json.dumps(curves, indent=1))

(HERE / "fuller.md").write_text("\n".join(lines) + "\n")
