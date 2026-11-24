"""The find term from titles alone, in the 4-round game: the collaborators' stage 1 logs and our own 4-round runs.

A seeker reads one page per round. With K other holders who posted the round's value, it finds the value if its
title is one of theirs. If copies choose titles independently given their role,

    find(K) = sum over titles t of  r_t * (1 - (1 - w_t)^K)

  r_t  share of seekers that read title t
  w_t  share of posting holders whose posts of the value, up to that round, include title t

With K = 1 this is the title match of the hackathon report (sum_t r_t w_t, its M when readers and writers share one
distribution). Nothing is fitted: r and w are counted on small swarms (N <= 8) and the larger swarms are predictions.
In sections 3 to 5 K is the number of posters counted in each target game, so those are predictions of finding given
posting. Section 6 forecasts K as well, from how often holders post in the small swarms.
Titles are compared as templates (the state's name replaced by <S>), as in the report.

  1. What the logs hold.
  2. Titles in round 1, by how many answers a copy holds.
  3. Find against the coverage term, round 1 and rounds 2+, every set.
  4. Equal holders: the stage 1 gap with each grid's own titles and with the other grid's.
  5. Mixed swarms: each model's seekers, from the titles of the pure swarms.
  6. The same predictions with the number of posters forecast from the small swarms (added 16 November 2026).

  python coverage.py [stage 1 dir] [benchmark dir]      (writes coverage.md)

The stage 1 logs are not in git; the default is 4-round-results/ at the repo root.
"""
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES  # noqa: E402
BENCH = ROOT
if not STAGE1.is_dir():
    sys.exit(f"stage 1 logs not found at {STAGE1}")
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))      # model client only; not needed here
sys.path.insert(0, str(BENCH))
from scb.io import read_jsonl  # noqa: E402

OURS = RES / "swarm4round"
NAME = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B", "qwen3-32b-bf16": "Qwen3-32B",
        "gemma-3-27b": "Gemma 3 27B"}
# key, label, what each copy holds, folders
SETS = [
    # Only folders named <prefix><N>: scaling/results also holds later runs such as swarm_comp_n4_gemma.
    ("main", "Stage 1, main grid", "2 of 4", sorted(d for d in STAGE1.glob("swarm_main_n*") if d.name[len("swarm_main_n"):].isdigit())),
    ("comp", "Stage 1, 1-of-4 grid", "1 of 4", sorted(d for d in STAGE1.glob("swarm_comp_n*") if d.name[len("swarm_comp_n"):].isdigit())),
    ("r1", "Stage 1, random halves", "random 2 of 4", sorted(d for d in STAGE1.glob("swarm_r1_n*") if d.name[len("swarm_r1_n"):].isdigit())),
    ("q8", "Stage 1, 8 questions", "4 of 8", sorted(d for d in STAGE1.glob("swarm_q8_n*") if d.name[len("swarm_q8_n"):].isdigit())),
    ("gemma", "Stage 1, Gemma", "2 of 4", sorted(d for d in STAGE1.glob("swarm_gemma_n*") if d.name[len("swarm_gemma_n"):].isdigit())),
    ("mix", "Stage 1, mixed swarms", "2 of 4", sorted(d for d in STAGE1.glob("swarm_mix_n*") if d.name[len("swarm_mix_n"):].isdigit())),
    ("ours", "Our 4-round runs", "2 of 4", sorted(OURS.glob("swarm_ref*")) if OURS.is_dir() else []),
]
SMALL = 8
rng = np.random.default_rng(11)
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def template(title, state):
    """A title with the state's name replaced, after the wiki's own normalisation (underscores, spaces, first letter)."""
    if title is None:
        return None
    t = " ".join(str(title).replace("_", " ").split())
    return re.sub(re.escape(state), "<S>", t[:1].upper() + t[1:], flags=re.I)


# ---------------------------------------------------------------------------------------------------------- read
chances = []                            # one row per chance
reads = defaultdict(Counter)            # (set, model, stage, N) -> read templates of seekers
pages = defaultdict(list)               # (set, model, stage, N) -> one set of templates per posting holder
games = Counter()                       # (set, mix label, N) -> games
gid = 0
for key, _, _, dirs in SETS:
    for d in dirs:
        for f in sorted((d / "raw").glob("*.jsonl*")):
            for r in read_jsonl(f):
                if "error" in r or r.get("awareness") != "unaware" or r.get("rule") != "forbidden":
                    continue
                gid += 1
                n, mixlab = r["n_agents"], r["model"]
                seat = r.get("seat_models") or {}
                model = {a: (seat.get(a) if isinstance(seat, dict) else None) or r["model"] for a in r["agents"]}
                if isinstance(seat, list):
                    model = dict(zip(sorted(r["agents"]), seat))
                games[(key, mixlab, n)] += 1
                hold = {a: set(v) for a, v in r["holds"].items()}
                posts = defaultdict(lambda: defaultdict(list))        # copy -> state -> [(round, title)]
                for a, ev in r["events"].items():
                    for w in ev["writes"]:
                        for st in w.get("posts", []):
                            posts[a][st].append((w["round"], w["path"]))
                read = {(a, x["round"]): x["path"] for a, ev in r["events"].items() for x in ev["reads"]}
                for a, ag in r["agents"].items():
                    for ans in ag["answers"]:
                        rd, st, stage = ans["round"], ans["state"], min(ans["round"], 2)
                        if st in hold[a]:
                            mine = frozenset(template(p, st) for w_, p in posts[a].get(st, []) if w_ <= rd)
                            if mine:
                                pages[(key, model[a], stage, n)].append(mine)
                            continue
                        if not ans["chance"]:
                            continue
                        K = Counter(model[b] for b in hold if b != a and st in hold[b]
                                    and any(w_ <= rd for w_, _ in posts[b].get(st, [])))
                        t = template(read.get((a, rd)), st)
                        reads[(key, model[a], stage, n)][t] += 1
                        chances.append(dict(set=key, mix=mixlab, model=model[a], N=n, stage=stage, game=gid, K=K,
                                            holders=sum(1 for b in hold if st in hold[b]), read=t,
                                            holders_by=Counter(model[b] for b in hold if st in hold[b]),
                                            seen=int(ans["seen"]), used=int(ans["used_other"])))


def read_dist(key, model, stage, ns=None):
    c = Counter()
    for (k, m, s, n), v in reads.items():
        if (k, m, s) == (key, model, stage) and (ns is None or n in ns):
            c.update(v)
    tot = sum(c.values())
    return {t: v / tot for t, v in c.items()} if tot else {}


def write_incl(key, model, stage, ns=None):
    sets = [s for (k, m, st, n), v in pages.items() if (k, m, st) == (key, model, stage) and (ns is None or n in ns)
            for s in v]
    c = Counter(t for s in sets for t in s)
    return {t: v / len(sets) for t, v in c.items()} if sets else {}


def coverage(r, ws, K):
    """1 - sum_t r_t * prod over poster models of (1 - w_t)^K_model."""
    miss = 0.0
    for t, p in r.items():
        q = 1.0
        for m, k in K.items():
            q *= (1 - ws[m].get(t, 0.0)) ** k
        miss += p * q
    return 1 - miss


def observed(rows, col="seen"):
    """Mean and 95% interval, resampling whole games."""
    per = defaultdict(lambda: [0, 0])
    for c in rows:
        per[c["game"]][0] += c[col]
        per[c["game"]][1] += 1
    a = np.array(list(per.values()), float)
    draws = [a[rng.integers(0, len(a), len(a))].sum(0) for _ in range(1000)]
    lo, hi = np.percentile([d[0] / d[1] for d in draws], [2.5, 97.5])
    return a[:, 0].sum() / a[:, 1].sum(), lo, hi


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def mix_name(label):
    """'qwen3-8b:2,gemma-3-27b:6' -> '2 Qwen3-8B + 6 Gemma 3 27B'; a single model's name otherwise."""
    if ":" not in label:
        return NAME.get(label, label)
    return " + ".join(f"{v} {NAME[k]}" for k, v in (p.split(":") for p in label.split(",")))


def small_ns(key, model):
    return sorted({n for (k, m, _, n) in reads if (k, m) == (key, model) and n <= SMALL})


pure = [(k, lab, held, m) for k, lab, held, _ in SETS if k != "mix"
        for m in sorted({c["model"] for c in chances if c["set"] == k}, key=list(NAME).index)]

say("# The find term from titles alone (4-round game)\n")
say("find(K) = Σ_t r_t (1 − (1 − w_t)^K): r_t is the share of seekers that read title t, w_t the share of posting "
    "holders whose posts of the value include it, K the number of other holders that posted it. Nothing is fitted: r "
    f"and w are counted on swarms of N ≤ {SMALL} and the larger swarms are predictions. With K = 1 this is the title "
    "match of the hackathon report. In sections 3 to 5 K is counted in each target game, so the coverage term "
    "predicts finding given posting; section 6 forecasts K too. Unaware text, rule in force. Rerun with "
    "`python coverage.py`.\n")

# ------------------------------------------------------------------------------------------------------------- 1
say("## 1. What the logs hold\n")
say("| Set | Model | Each copy holds | Swarm sizes | Games per size |")
say("|---|---|---|---|---|")
for key, lab, held, _ in SETS:
    by = defaultdict(dict)
    for (k, mixlab, n), g in games.items():
        if k == key:
            by[mixlab][n] = g
    for mixlab, d in sorted(by.items(), key=lambda kv: list(NAME).index(kv[0]) if kv[0] in NAME else 99):
        say(f"| {lab} | {mix_name(mixlab)} | {held} | {', '.join(map(str, sorted(d)))} | "
            f"{', '.join(map(str, sorted(set(d.values()))))} |")

# ------------------------------------------------------------------------------------------------------------- 2
say("\n## 2. Titles in round 1, by how many answers a copy holds\n")
say(f"Round 1 is before any copy has seen another's page. Counted on N ≤ {SMALL}. Bare state name: the seeker read the "
    "page titled with the state alone, where holders almost never post. Title match: Σ_t r_t w_t, the chance that "
    "one seeker and one posting holder meet.\n")
say("| Set | Model | Each copy holds | Seekers' most-read title | Its share | Seekers reading the bare state name | "
    "Holders' top title, share | Title match |")
say("|---|---|---|---|---|---|---|---|")
for key, lab, held, m in pure:
    ns = small_ns(key, m)
    r, w = read_dist(key, m, 1, ns), write_incl(key, m, 1, ns)
    if not r or not w:
        continue
    top = max((t for t in r if t), key=lambda t: r[t])
    say(f"| {lab} | {NAME[m]} | {held} | {top} | {pct(r[top])}% | {pct(r.get('<S>', 0))}% | {pct(max(w.values()))}% | "
        f"{pct(sum(p * w.get(t, 0) for t, p in r.items()))}% |")

# ------------------------------------------------------------------------------------------------------------- 3
say("\n## 3. Find against the coverage term\n")
say(f"Share of chances in which the seeker read a page holding the value. Coverage term: titles from N ≤ {SMALL} of the "
    "same set, model and stage (round 1, or rounds 2 and later pooled), applied to each chance's K. Cells above "
    f"N = {SMALL} are held out. Observed: 95% interval from resampling games.\n")
say("| Set | Model | Stage | N | Mean K | Observed find | Coverage term | Held out |")
say("|---|---|---|---|---|---|---|---|")
errs = []
obs3 = {}                               # (set, model, stage, N) -> observed find and its interval, kept for section 6
for key, lab, held, m in pure:
    ns = small_ns(key, m)
    for stage, slab in ((1, "round 1"), (2, "rounds 2+")):
        r, w = read_dist(key, m, stage, ns), write_incl(key, m, stage, ns)
        if not r or not w:
            continue
        for n in sorted({c["N"] for c in chances if c["set"] == key and c["model"] == m}):
            rows = [c for c in chances if (c["set"], c["model"], c["N"], c["stage"]) == (key, m, n, stage)]
            o, lo, hi = observed(rows)
            obs3[(key, m, stage, n)] = (o, lo, hi)
            pred = np.mean([coverage(r, {m: w}, c["K"]) for c in rows])
            held_out = n > SMALL
            if held_out:
                errs.append(dict(set=key, model=m, stage=stage, err=100 * (pred - o), inside=lo <= pred <= hi))
            say(f"| {lab} | {NAME[m]} | {slab} | {n} | {np.mean([sum(c['K'].values()) for c in rows]):.1f} | "
                f"{pct(o)} [{pct(lo)}, {pct(hi)}] | {pct(pred)} | {'yes' if held_out else ''} |")
say("\n| Held-out cells | Cells | Median error | Largest error | Mean signed error | Inside the observed interval |")
say("|---|---|---|---|---|---|")
groups = [("Qwen3-8B, round 1 (stage 1 and ours)", lambda e: e["model"] == "qwen3-8b" and e["stage"] == 1),
          ("Qwen3-8B, rounds 2+", lambda e: e["model"] == "qwen3-8b" and e["stage"] == 2),
          ("Qwen3-4B, 14B, 32B (ours), round 1", lambda e: e["model"] != "qwen3-8b" and e["stage"] == 1),
          ("Qwen3-4B, 14B, 32B (ours), rounds 2+", lambda e: e["model"] != "qwen3-8b" and e["stage"] == 2)]
for lab, sel in groups:
    g = [e for e in errs if sel(e)]
    if g:
        a = np.array([e["err"] for e in g])
        say(f"| {lab} | {len(g)} | {np.median(np.abs(a)):.1f} | {np.abs(a).max():.1f} | {a.mean():+.1f} | "
            f"{sum(e['inside'] for e in g)} of {len(g)} |")
say("\nErrors in points (coverage term minus observed).")

# ------------------------------------------------------------------------------------------------------------- 4
say("\n## 4. Equal holders: the stage 1 gap\n")
say("Stage 1 compared swarms with the same number of holders per answer: each copy holding 2 of 4 answers, or 1 of 4 "
    "in a swarm twice the size. Round 1 find, with the coverage term given each grid's own titles and the other "
    f"grid's (all from N ≤ {SMALL}).\n")
say("| Holders per answer | Each copy holds | N | CR-v2, all rounds | Find, round 1 | Coverage, own titles | "
    "Coverage, the other grid's titles |")
say("|---|---|---|---|---|---|---|")
m8 = "qwen3-8b"
T = {k: (read_dist(k, m8, 1, small_ns(k, m8)), write_incl(k, m8, 1, small_ns(k, m8))) for k in ("main", "comp")}
for h in (2, 4, 8):
    for key, held, n in (("main", "2 of 4", 2 * h), ("comp", "1 of 4", 4 * h)):
        allr = [c for c in chances if (c["set"], c["N"]) == (key, n)]
        r1 = [c for c in allr if c["stage"] == 1]
        other = "comp" if key == "main" else "main"
        own = np.mean([coverage(T[key][0], {m8: T[key][1]}, c["K"]) for c in r1])
        oth = np.mean([coverage(T[other][0], {m8: T[other][1]}, c["K"]) for c in r1])
        say(f"| {h} | {held} | {n} | {pct(observed(allr, 'used')[0])} | {pct(observed(r1)[0])} | {pct(own)} | {pct(oth)} |")
say("\nWhich side of the titles carries the gap, in the 1-of-4 grid (coverage term, round 1):\n")
say("| N | Observed | Reads and writes of the 1-of-4 grid | Its reads, main-grid writes | Main-grid reads, its writes | "
    "Reads and writes of the main grid |")
say("|---|---|---|---|---|---|")
for n in (8, 16, 32):
    r1 = [c for c in chances if (c["set"], c["N"], c["stage"]) == ("comp", n, 1)]
    cells = [np.mean([coverage(T[a][0], {m8: T[b][1]}, c["K"]) for c in r1])
             for a, b in (("comp", "comp"), ("comp", "main"), ("main", "comp"), ("main", "main"))]
    say(f"| {n} | {pct(observed(r1)[0])} | " + " | ".join(pct(x) for x in cells) + " |")

# ------------------------------------------------------------------------------------------------------------- 5
say("\n## 5. Mixed swarms\n")
say("Qwen3-8B and Gemma 3 27B on one wiki. Each model's seekers, with the coverage term built from the pure swarms' "
    "titles (Qwen from the main grid, Gemma from its own runs; N ≤ 8): 1 − Σ_t r_t Π_model (1 − w_t)^K_model. "
    "The titles come from the pure swarms only. K_model, the number of posters of each model, is counted in each "
    "mixed game; section 6 forecasts it from the pure swarms too.\n")
say("| Swarm | Seekers | Stage | Observed find | Coverage term from the pure swarms | Chances |")
say("|---|---|---|---|---|---|")
src = {"qwen3-8b": "main", "gemma-3-27b": "gemma"}
mix_err = []
obs5 = {}                               # (mix, model, stage) -> observed find and its interval, kept for section 6
for mixlab in sorted({c["mix"] for c in chances if c["set"] == "mix"}):
    parts = dict(p.split(":") for p in mixlab.split(","))
    name = mix_name(mixlab)
    for m in parts:
        for stage, slab in ((1, "round 1"), (2, "rounds 2+")):
            rows = [c for c in chances if c["set"] == "mix" and c["mix"] == mixlab and c["model"] == m and c["stage"] == stage]
            if not rows:
                continue
            r = read_dist(src[m], m, stage, small_ns(src[m], m))
            ws = {k: write_incl(src[k], k, stage, small_ns(src[k], k)) for k in parts}
            o, lo, hi = observed(rows)
            obs5[(mixlab, m, stage)] = (o, lo, hi)
            pred = np.mean([coverage(r, ws, c["K"]) for c in rows])
            mix_err.append((stage, 100 * (pred - o), lo <= pred <= hi))
            say(f"| {name} | {NAME[m]} | {slab} | {pct(o)} [{pct(lo)}, {pct(hi)}] | {pct(pred)} | {len(rows):,} |")
for stage, slab in ((1, "Round 1"), (2, "Rounds 2+")):
    a = np.array([e for s, e, _ in mix_err if s == stage])
    if len(a):
        say(f"\n{slab}: median error {np.median(np.abs(a)):.1f} points, largest {np.abs(a).max():.1f}; inside the observed "
            f"interval in {sum(i for s, _, i in mix_err if s == stage)} of {len(a)} cells.")

# ------------------------------------------------------------------------------------------------------------- 6
# Added 16 November 2026, after review. No random draw is made here: the observed intervals are those of sections 3
# and 5, so nothing above moves.
say("\n## 6. With the number of posters forecast as well\n")
say("Sections 3 to 5 give the coverage term each target game's own K. Here K is forecast too. Each holder of the "
    f"answer posts with chance π, the share of holders that had posted in swarms of N ≤ {SMALL} of the same set, model "
    "and stage, independently of the others. The mean of (1 − w_t)^K over K ~ Binomial(h, π) is (1 − π w_t)^h, so "
    "the forecast is 1 − Σ_t r_t Π_model (1 − π_model w_t)^h_model, with h the holders of the answer. Nothing from "
    "the target cells is used.\n")


def post_rate(key, m, stage):
    """Share of the other holders of an answer that had posted it, over the small swarms' chances."""
    rows = [c for c in chances if (c["set"], c["model"], c["stage"]) == (key, m, stage) and c["N"] in small_ns(key, m)
            and set(c["holders_by"]) <= {m}]
    den = sum(c["holders"] for c in rows)
    return sum(sum(c["K"].values()) for c in rows) / den if den else float("nan")


def forecast(r, ws, pis, hb):
    miss = 0.0
    for t, p in r.items():
        q = 1.0
        for m, h in hb.items():
            q *= (1 - pis[m] * ws[m].get(t, 0.0)) ** h
        miss += p * q
    return 1 - miss


say("| Set | Model | Stage | N | π, small swarms | Posting in the cell | Observed find | Given K | K forecast |")
say("|---|---|---|---|---|---|---|---|---|")
f_err = []
for key, lab, held, m in pure:
    ns = small_ns(key, m)
    for stage, slab in ((1, "round 1"), (2, "rounds 2+")):
        r, w = read_dist(key, m, stage, ns), write_incl(key, m, stage, ns)
        if not r or not w:
            continue
        pi = post_rate(key, m, stage)
        for n in sorted({c["N"] for c in chances if c["set"] == key and c["model"] == m and c["N"] > SMALL}):
            rows = [c for c in chances if (c["set"], c["model"], c["N"], c["stage"]) == (key, m, n, stage)]
            o, lo, hi = obs3[(key, m, stage, n)]
            given = np.mean([coverage(r, {m: w}, c["K"]) for c in rows])
            fc = np.mean([forecast(r, {m: w}, {m: pi}, c["holders_by"]) for c in rows])
            there = sum(sum(c["K"].values()) for c in rows) / sum(c["holders"] for c in rows)
            f_err.append(dict(model=m, stage=stage, given=100 * (given - o), fc=100 * (fc - o),
                              in_given=lo <= given <= hi, in_fc=lo <= fc <= hi))
            say(f"| {lab} | {NAME[m]} | {slab} | {n} | {pct(pi)} | {pct(there)} | {pct(o)} [{pct(lo)}, {pct(hi)}] | "
                f"{pct(given)} | {pct(fc)} |")
say("\n| Held-out cells | Cells | Given K: median, largest error | Inside | K forecast: median, largest error | Inside |")
say("|---|---|---|---|---|---|")
for lab, sel in groups + [("Qwen3-14B (ours), round 1", lambda e: e["model"] == "qwen3-14b" and e["stage"] == 1)]:
    g = [e for e in f_err if sel(e)]
    if g:
        a, b = np.abs([e["given"] for e in g]), np.abs([e["fc"] for e in g])
        say(f"| {lab} | {len(g)} | {np.median(a):.1f}, {a.max():.1f} | {sum(e['in_given'] for e in g)} of {len(g)} | "
            f"{np.median(b):.1f}, {b.max():.1f} | {sum(e['in_fc'] for e in g)} of {len(g)} |")
say("\nErrors in points. For Qwen3-8B in round 1 the forecast of K costs nothing. For Qwen3-14B it does. The mean K "
    "is the same either way (the two posting columns), so the difference is in how the posters are spread: in the "
    "games they are bunched on some questions, and independent holders would spread them over more.\n")

say("Mixed swarms, with each model's π from its pure swarms:\n")
say("| Swarm | Seekers | Stage | Observed find | Given K | K forecast |")
say("|---|---|---|---|---|---|")
m_err = []
for mixlab in sorted({c["mix"] for c in chances if c["set"] == "mix"}):
    parts = dict(p.split(":") for p in mixlab.split(","))
    for m in parts:
        for stage, slab in ((1, "round 1"), (2, "rounds 2+")):
            rows = [c for c in chances if c["set"] == "mix" and c["mix"] == mixlab and c["model"] == m and c["stage"] == stage]
            if not rows:
                continue
            r = read_dist(src[m], m, stage, small_ns(src[m], m))
            ws = {k: write_incl(src[k], k, stage, small_ns(src[k], k)) for k in parts}
            pis = {k: post_rate(src[k], k, stage) for k in parts}
            o, lo, hi = obs5[(mixlab, m, stage)]
            given = np.mean([coverage(r, ws, c["K"]) for c in rows])
            fc = np.mean([forecast(r, ws, pis, c["holders_by"]) for c in rows])
            m_err.append((stage, 100 * (given - o), 100 * (fc - o), lo <= given <= hi, lo <= fc <= hi))
            say(f"| {mix_name(mixlab)} | {NAME[m]} | {slab} | {pct(o)} [{pct(lo)}, {pct(hi)}] | {pct(given)} | {pct(fc)} |")
for stage, slab in ((1, "Round 1"), (2, "Rounds 2+")):
    g = [e for e in m_err if e[0] == stage]
    if g:
        a, b = np.abs([e[1] for e in g]), np.abs([e[2] for e in g])
        say(f"\n{slab}: given K, median error {np.median(a):.1f} points, largest {a.max():.1f}, inside the observed "
            f"interval in {sum(e[3] for e in g)} of {len(g)} cells; K forecast, median {np.median(b):.1f}, largest "
            f"{b.max():.1f}, inside in {sum(e[4] for e in g)} of {len(g)}.")

(HERE / "coverage.md").write_text("\n".join(lines) + "\n")
