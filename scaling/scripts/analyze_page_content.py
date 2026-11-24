#!/usr/bin/env python3
"""What on a crowded page makes a late holder post: answers or requests, whether they help, and their wording.

  .venv/bin/python scripts/analyze_page_content.py results [--source aaai]
      writes results/page_content.json and prints, per model and arm, how often late holders post their round-2
      answer, and the tests registered in results/predictions_page_content.md; with --source, also the observational
      check that registration starts from

The design and the forecasts are registered in results/predictions_page_content.md, written before the runs. Pairs of
Gemma 3 27B (tag plant_content_gemma) and of Qwen3-8B (plant_content_q8) on the fixed prompt. Before round 1, nothing
or one of the pages written by scripts/make_content_pages.py (results/page_content/) sits on the pair's usual round-1
title for question 1. A late holder first holds an answer in round 2 (in a pair, copy A). Treated: its round-1 read
was the planted title (with nothing planted, the same title). Posting: a round-2 write carrying the round-2 state's
true value. Intervals resample seeds, all games of a seed, in every arm of both models, moving together (an arm's
pages, and both models' quizzes, are the same for a seed).

Gemma's two arms of answers about another state (answers_unasked, answers_q3) plant the same notes about a state that
is not asked by round 2, so a late holder cannot tell them apart before round 3: for round-2 posting they are pooled
as answers_other_state (Qwen3-8B plays answers_unasked only). Seeds where question 3's value equals question 1's are
left out of answers_q3, and seeds where question 2's value equals question 1's out of answers_q2 (both models): there
the planted notes also carry question 1's value (3 and 4 of the 100 seeds; scripts/make_content_pages.py lists them).

With --source (the folder holding the aaai branch's result folders, as for make_content_pages.py), it also repeats the
observational check: in the swarms already run, within each swarm size, a late holder's posting against the answers
and the requests on the page it read in round 1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_transplant import Arms, late_holder, rows_of  # noqa: E402
from title_law import condition, fixed, load, pct  # noqa: E402

MODELS = {"gemma-3-27b": ("plant_content_gemma", "gemma", "{state} Construction Employment Share"),
          "qwen3-8b": ("plant_content_q8", "q8", "{state} construction occupation share")}
POOLED = {"answers_other_state": ("answers_unasked", "answers_q3")}
ARMS = ("nothing", "answers_q1", "answers_other_state", "answers_unasked", "answers_q3", "answers_q1_other",
        "requests_q1", "requests_q2", "answers_q2")
# registered (predictions_page_content.md): label -> (model, arm, arm it is compared with, sides, the test)
REGISTERED = {
    "a": ("gemma-3-27b", "answers_q1", "answers_other_state", 1,
          "reciprocity: answers that help the late holder in round 1 raise posting more than the same answers about "
          "another state"),
    "b": ("gemma-3-27b", "answers_other_state", "nothing", 1,
          "norm: answers that do not help the late holder still raise posting"),
    "c": ("gemma-3-27b", "answers_q1", "answers_q1_other", 2,
          "wording: answers to question 1 in Gemma's own words and as bare numbers differ"),
    "d": ("qwen3-8b", "requests_q2", "requests_q1", 1,
          "targeted demand: requests for the late holder's own question raise posting more than requests for "
          "question 1"),
    "e": ("qwen3-8b", "nothing", "answers_q2", 1,
          "redundancy: the late holder's own answer, already on the page, lowers posting"),
    "f": ("qwen3-8b", "answers_unasked", "answers_q2", 1,
          "redundancy of its own answer: its own answer lowers posting more than the same answers about another state"),
}
# reported, not registered: (arm, compared with, outcome)
REPORTED = [("answers_q1", "nothing", "post2"), ("answers_other_state", "nothing", "post2"),
            ("answers_unasked", "nothing", "post2"), ("answers_q1", "answers_unasked", "post2"),
            ("answers_unasked", "answers_q3", "post2"), ("answers_q3", "answers_unasked", "post4"),
            ("requests_q1", "nothing", "post2"), ("requests_q2", "nothing", "post2"),
            ("requests_q2", "requests_q1", "post2"), ("answers_q2", "nothing", "post2"),
            ("answers_other_state", "answers_q2", "post2"), ("answers_unasked", "answers_q2", "post2"),
            ("answers_q1", "answers_q1_other", "post2")]
TIE_ARMS = {"answers_q3": 2, "answers_q2": 1}      # arm -> the question whose value it plants
BARE = re.compile(r"^\s*(?:[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*\s*:\s*)?\d+(?:\.\d+)?\s*%?\s*\.?\s*$")


def arm(r: dict, short: str) -> str:
    p = r.get("plant")
    return p["file"].removeprefix(f"{short}_") if isinstance(p, dict) and "file" in p else "nothing"


def tied(r: dict, k: int) -> bool:
    """Question k's value (0-based) equals question 1's: a note giving it also carries question 1's value."""
    truth = {x["state"]: x["truth"] for ag in r["agents"].values() for x in ag["answers"]}
    qs = r["questions"]
    return f"{truth[qs[k]]:.1f}" == f"{truth[qs[0]]:.1f}"


def note_text(r: dict, name: str, rnd: int) -> str:
    """The text a copy wrote in round `rnd` (its transcript: system, then six messages a round, the write third)."""
    raw = r["agents"][name]["messages"][2 + 6 * (rnd - 1)]["content"]
    try:
        return str(json.loads(raw).get("edit", {}).get("text", ""))
    except (ValueError, AttributeError):
        m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        return m.group(1) if m else raw


def late_record(r: dict, title: str) -> dict | None:
    """late_holder(), with whether a round-2 post is a bare number ("<value>" or "<State>: <value>", the form of the
    planted answers in Gemma's other wording)."""
    x = late_holder(r, title)
    if x is None:
        return None
    qs = r["questions"]
    n = next(a for a in r["events"] if qs[0] not in r["holds"][a] and qs[1] in r["holds"][a])
    return x | {"bare2": bool(BARE.match(note_text(r, n, 2))) if x["post2"] else None,
                "offset": r.get("sample_offset", 0)}


def observe(source: Path, rng: np.random.Generator, reps: int) -> dict:
    """Within swarm size: late holders' posting on log2(1 + answers) and log2(1 + requests) on the page they read."""
    sc = Path(__file__).resolve().parent.parent
    rows = load(sc / "results", [sc.parent / "wiki_game" / "data" / "swarm4round", source])
    rows = [r for r in rows if r["model"] in MODELS and condition(r) == fixed()]
    have = defaultdict(set)
    for r in rows:
        have[(r["model"], r["n_agents"])].add(r["_src"])
    rows = [r for r in rows if r["_src"] == ("results" if "results" in have[(r["model"], r["n_agents"])]
                                               else r["_src"])]          # one source per model and size
    out = {}
    for model in MODELS:
        games = defaultdict(list)
        for r in (x for x in rows if x["model"] == model):
            qs, ev, holds = r["questions"], r["events"], r["holds"]
            w1 = {a: next((w for w in ev[a]["writes"] if w["round"] == 1), None) for a in ev}
            top = Counter(w["path"] for w in w1.values() if w and w.get("path")).most_common(1)[0][0]
            recs = []
            for late in ev:
                if qs[0] in holds[late] or qs[1] not in holds[late]:
                    continue
                rd = next((x for x in ev[late]["reads"] if x["round"] == 1 and x.get("path")), None)
                if rd is None:
                    continue
                on = [(a, w) for a, w in w1.items() if a != late and w and w.get("path") == rd["path"]]
                n_a = sum(qs[0] in holds[a] and qs[0] in w["posts"] for a, w in on)
                n_r = sum(qs[0] not in holds[a] and not w["posts"] for a, w in on)
                y = any(w["round"] == 2 and qs[1] in w["posts"] for w in ev[late]["writes"])
                recs.append((n_a, n_r, rd["path"] == top, y))
            games[r["n_agents"]].append(recs)
        sizes = sorted(games)

        def fit(sample: dict) -> np.ndarray:
            X, Y = [], []
            for n, gl in sample.items():
                for g in gl:
                    for n_a, n_r, _, y in g:
                        X.append([1.0 * (n == s) for s in sizes] + [np.log2(1 + n_a), np.log2(1 + n_r)])
                        Y.append(y)
            X, Y = np.array(X), np.array(Y, float)
            keep = np.r_[X[:, :len(sizes)].sum(0) > 0, True, True]
            return np.linalg.lstsq(X[:, keep], Y, rcond=None)[0][-2:]

        b = fit(games)
        boots = np.array([fit({n: [gl[i] for i in rng.integers(0, len(gl), len(gl))] for n, gl in games.items()})
                          for _ in range(reps)])
        recs = [x for gl in games.values() for g in gl for x in g]
        la, lr = np.log2(1 + np.array([x[0] for x in recs])), np.log2(1 + np.array([x[1] for x in recs]))
        big = [x for g in games.get(64, []) for x in g]
        split = {lab: {"late_holders": len(g), "median_notes": float(np.median([x[0] + x[1] for x in g])),
                       "main_title": float(np.mean([x[2] for x in g])), "post2": float(np.mean([x[3] for x in g]))}
                 for lab, g in (("answers outnumber requests", [x for x in big if x[0] > x[1]]),
                                ("otherwise", [x for x in big if x[0] <= x[1]])) if g}
        out[model] = {"sizes": sizes, "late_holders": len(recs), "per_doubling_answers": float(b[0]),
                      "per_doubling_answers_ci": pct(list(boots[:, 0])), "per_doubling_requests": float(b[1]),
                      "per_doubling_requests_ci": pct(list(boots[:, 1])), "correlation": float(np.corrcoef(la, lr)[0, 1]),
                      "n64_split": split}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--source", help="folder holding the aaai branch's result folders (for the observational check)")
    ap.add_argument("--tags", nargs=2, default=None, help="the two tags (Gemma, Qwen3-8B) if not the registered ones")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="page_content.json")
    a = ap.parse_args()
    res = Path(a.results)
    out: dict = {}
    if a.source:
        obs = out["observational"] = observe(Path(a.source), np.random.default_rng(0), min(a.reps, 1000))
        print("== the swarms already run: late holders' posting per doubling of the notes on their round-1 page")
        for model, x in obs.items():
            print(f"   {model}: answers {100 * x['per_doubling_answers']:+.1f} [{100 * x['per_doubling_answers_ci'][0]:+.1f}, "
                  f"{100 * x['per_doubling_answers_ci'][1]:+.1f}], requests {100 * x['per_doubling_requests']:+.1f} "
                  f"[{100 * x['per_doubling_requests_ci'][0]:+.1f}, {100 * x['per_doubling_requests_ci'][1]:+.1f}] points; "
                  f"correlation of the two counts {x['correlation']:.2f}")
            for lab, s in x["n64_split"].items():
                print(f"      64 copies, {lab}: {s['late_holders']} late holders, median {s['median_notes']:.0f} notes, "
                      f"{100 * s['main_title']:.1f}% on the main title, {100 * s['post2']:.1f}% post")
    tags = dict(zip(MODELS, a.tags)) if a.tags else {m: v[0] for m, v in MODELS.items()}
    arms: dict[str, Arms] = {}
    for model, (_, short, usual) in MODELS.items():
        folder = res / f"swarm_{tags[model]}"
        if not folder.is_dir():
            continue
        recs: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows_of(folder):
            if r["model"] != model or r["n_agents"] != 2 or condition(r) != fixed():
                continue
            k = arm(r, short)
            if k in TIE_ARMS and tied(r, TIE_ARMS[k]):
                continue
            p = r.get("plant")
            x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
            if x:
                recs[k][r["seed"]].append(x)
        for pooled, parts in POOLED.items():
            if all(k in recs for k in parts):
                for k in parts:
                    for s, xs in recs[k].items():
                        recs[pooled][s] += xs
        arms[model] = Arms(recs)
    # one resample of the seeds for both models: the same seed is the same quiz (and the same source game) in both
    seeds = sorted(set().union(*[set(A.seeds) for A in arms.values()])) if arms else []
    rng = np.random.default_rng(0)
    boots = [rng.choice(seeds, size=len(seeds)) for _ in range(a.reps)]
    out["models"], out["registered"], out["between_models"] = {}, {}, {}
    for model, A in arms.items():
        recs = A.recs
        print(f"== {model} ({tags[model]}): late holders that read the planted title in round 1")
        print(f"   {'arm':20s} {'games':>5s} {'read it':>7s} {'post, round 2 [95%]':>22s} {'all late holders':>17s} "
              f"{'post, round 4':>14s} {'took the plant':>15s} {'bare posts':>11s}")
        m = out["models"][model] = {"arms": {}, "differences": {}}
        for k in (k for k in ARMS if k in recs):
            xs = [x for v in recs[k].values() for x in v]
            p, ci = A.rate(k), pct([A.rate(k, b) for b in boots])
            posted = [x["bare2"] for x in xs if x["treated"] and x["post2"]]
            m["arms"][k] = {"games": len(xs), "treated": sum(x["treated"] for x in xs), "post2": p, "post2_ci": ci,
                            "post2_all": A.rate(k, treated=False), "post4": A.rate(k, key="post4"),
                            "took_plant": A.rate(k, key="took_plant"),
                            "bare_share_of_posts": float(np.mean(posted)) if posted else float("nan")}
            row = m["arms"][k]
            print(f"   {k:20s} {row['games']:5d} {row['treated']:7d} {100 * p:9.1f} [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]"
                  f"  {100 * row['post2_all']:15.1f}  {100 * row['post4']:13.1f}  {100 * row['took_plant']:14.1f}"
                  f"  {100 * row['bare_share_of_posts']:10.1f}")
        pairs = [(hi, lo, "post2") for mod, hi, lo, _, _ in REGISTERED.values() if mod == model] + REPORTED
        for hi, lo, key in dict.fromkeys(pairs):
            if hi in recs and lo in recs:
                f = lambda s, hi=hi, lo=lo, key=key: A.rate(hi, s, key=key) - A.rate(lo, s, key=key)   # noqa: E731
                d, ci = f(None), pct([f(b) for b in boots])
                name = f"{hi} - {lo}" + ("" if key == "post2" else f" ({key})")
                m["differences"][name] = {"d": d, "ci": ci}
                print(f"   {name:46s} {100 * d:+6.1f} [{100 * ci[0]:+6.1f}, {100 * ci[1]:+6.1f}] points")
        for lab, (mod, hi, lo, sides, text) in REGISTERED.items():
            if mod == model and f"{hi} - {lo}" in m["differences"]:
                g = m["differences"][f"{hi} - {lo}"]
                met = g["ci"][0] > 0 or (sides == 2 and g["ci"][1] < 0)
                out["registered"][lab] = {"test": text, "sides": sides, **g, "met": bool(met)}
                print(f"   registered ({lab}) {text}: {'met' if met else 'not met'}")
    if len(arms) == 2 and all({"answers_q2", "nothing"} <= set(A.recs) for A in arms.values()):
        G, Q = arms["gemma-3-27b"], arms["qwen3-8b"]
        f = lambda s: (G.rate("answers_q2", s) - G.rate("nothing", s)) - (Q.rate("answers_q2", s) - Q.rate("nothing", s))  # noqa: E731
        d, ci = f(None), pct([f(b) for b in boots])
        out["between_models"]["answers_q2 - nothing, Gemma minus Qwen3-8B"] = {"d": d, "ci": ci}
        print(f"== the late holder's own answer already on the page (answers_q2 - nothing), Gemma minus Qwen3-8B: "
              f"{100 * d:+.1f} [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] points")
    (res / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
