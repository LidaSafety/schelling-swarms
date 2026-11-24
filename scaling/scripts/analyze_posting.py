#!/usr/bin/env python3
"""Posting and wrong numbers: why swarms depart from the title law (scripts/title_law.py).

  .venv/bin/python scripts/analyze_posting.py results [--extra ../wiki_game/data/swarm4round]
      writes results/posting.json and prints one line per cell

Per cell (the fixed prompt; 2 answers each and 4 questions unless named):
  posted       share of (holder, question) where the holder posted the right value on a chosen title by the question's
               round (copies post in the question's own round, never earlier)
  wrong notes  share of non-holders' rounds in which they wrote a note naming the round's state with a wrong number
               (a note naming no other quiz state, holding a number with one decimal, not the true value)
  found        share of chances where the seeker had seen the right value (CR-v2's "seen")
  beside wrong share of found values whose page also showed a wrong number for that state
  uptake       used / found, overall, on clean pages, and on pages that also showed a wrong number
  law(base)    the title law from the family's smallest runs; law(cell) the same law measured on the cell's own runs
               (other runs only). law(cell) - law(base): how the title spread itself moved with N.
               measured - law(cell): what independent title picks inside the cell do not explain.
Per round (Qwen3-8B, 2 answers each): found, measured and predicted by both laws, for each question round.
Rule off (the ruleoff tags, --rules silent): the same rates with and without the rule, on the quizzes (seeds) both
  cells ran, and off minus on with 95% intervals from resampling those quizzes. The rule-on cell is this folder's,
  else the --extra folder's.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from title_law import Base, cells, condition, fixed, found, load, predict, target_chances  # noqa: E402
from scb.deaddrop import _channel_action, _mentions, is_placeholder  # noqa: E402
from scb.llm import extract_json  # noqa: E402

DEC = re.compile(r"(?<![\d.])\d{1,3}\.\d(?!\d)")


def notes_of(r: dict) -> list[tuple[int, str, str, str]]:
    """(round, author, title, text) of every note, parsed from the copies' WRITE replies."""
    out = []
    for n, ag in r["agents"].items():
        rnd, msgs = None, ag["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "user":
                continue
            g = re.match(r"ROUND (\d+)/", m["content"])
            rnd = int(g.group(1)) if g else rnd
            if "WRITE step." in m["content"] and i + 1 < len(msgs):
                act = _channel_action(r["channel"], extract_json(msgs[i + 1]["content"]) or {}, "write")
                if act is not None:
                    out.append((rnd, n, *act))
    return out


def judge(title: str, text: str, st: str, truth: dict) -> str | None:
    """'right' / 'wrong' / 'none' for a note about state st; None if the note does not name st."""
    low = (title + " " + text).lower()
    if st.lower() not in low:
        return None
    if _mentions(title, text, st, truth[st]):
        return "right"
    return "wrong" if DEC.search(text) and not any(s != st and s.lower() in low for s in truth) else "none"


def run_counts(r: dict) -> Counter:
    """The counts behind cell_stats, for one run."""
    c = Counter()
    truth = {a["state"]: a["truth"] for ag in r["agents"].values() for a in ag["answers"]}
    qs, names, ev = r["questions"], list(r["events"]), r["events"]
    holds = {n: set(r["holds"][n]) for n in names}
    notes = notes_of(r)
    by_title = defaultdict(list)
    for rk, au, t, x in notes:
        by_title[t].append((rk, au, t, x))
    for n in names:
        for k, st in enumerate(qs, 1):
            if st in holds[n]:
                c["h"] += 1
                c["h_post"] += any(st in w["posts"] and w["round"] <= k and not is_placeholder(w["path"])
                                   for w in ev[n]["writes"])
    for k, st in enumerate(qs, 1):
        c["nh"] += sum(st not in holds[n] for n in names)
        c["nh_wrong"] += sum(rk == k and st not in holds[au] and judge(t, x, st, truth) == "wrong"
                             for rk, au, t, x in notes)
    for n in names:
        reads = {e["round"]: e["path"] for e in ev[n]["reads"]}
        for a in r["agents"][n]["answers"]:
            if not a["chance"]:
                continue
            c["ch"] += 1
            if not a["seen"]:
                continue
            st, conf = a["state"], False
            for s in range(1, a["round"] + 1):
                js = [judge(t, x, st, truth) for rk, au, t, x in by_title.get(reads.get(s), []) if rk <= s and au != n]
                conf |= "right" in js and "wrong" in js
            u = a.get("used_chosen", a["used_other"])
            c["seen"] += 1
            c["used"] += u
            c["conf"] += conf
            c["used_conf"] += u and conf
    return c


def rates(c: Counter) -> dict:
    d = lambda a, b: c[a] / c[b] if c[b] else None
    clean = c["seen"] - c["conf"]
    return {"posted": d("h_post", "h"), "wrong_notes": d("nh_wrong", "nh"), "found": d("seen", "ch"),
            "beside_wrong": d("conf", "seen"), "uptake": d("used", "seen"), "cr": d("used", "ch"),
            "uptake_clean": (c["used"] - c["used_conf"]) / clean if clean else None,
            "uptake_beside_wrong": d("used_conf", "conf")}


def cell_stats(rows: list[dict]) -> dict:
    return {"runs": len(rows), **rates(sum((run_counts(r) for r in rows), Counter()))}


RULE_KEYS = ("posted", "wrong_notes", "found", "uptake", "cr")


def rule_off(rows: list[dict], reps: int = 1000, seed: int = 0) -> list[dict]:
    """The fixed prompt without the rule (--rules silent) against the same model, size and quizzes with the rule:
    rates on the quizzes both have, and off minus on with 95% intervals from resampling those quizzes (paired)."""
    on_cond = fixed()
    off_cond = on_cond[:4] + ("silent",) + on_cond[5:]
    out = []
    for m in sorted({r["model"] for r in rows if condition(r) == off_cond}):
        on_cells = cells(rows, m, on_cond)
        for (src, n), g_off in cells(rows, m, off_cond).items():
            g_on = on_cells.get(("results", n)) or next((v for (s, nn), v in on_cells.items() if nn == n), None)
            if not g_on:
                continue
            con, coff = {r["seed"]: run_counts(r) for r in g_on}, {r["seed"]: run_counts(r) for r in g_off}
            common = sorted(set(con) & set(coff))
            tot = lambda cs, ss: rates(sum((cs[s] for s in ss), Counter()))
            on_v, off_v = tot(con, common), tot(coff, common)
            rng, boot = random.Random(seed), defaultdict(list)
            for _ in range(reps):
                ss = [common[rng.randrange(len(common))] for _ in common]
                a, b = tot(con, ss), tot(coff, ss)
                for k in RULE_KEYS:
                    if a[k] is not None and b[k] is not None:
                        boot[k].append(b[k] - a[k])
            x = {"model": m, "N": n, "source_on": next(s for (s, nn), v in on_cells.items() if v is g_on),
                 "paired": len(common)}
            for k in RULE_KEYS:
                v = sorted(boot[k])
                x[k] = {"on": on_v[k], "off": off_v[k],
                        "diff": None if on_v[k] is None or off_v[k] is None else off_v[k] - on_v[k],
                        "ci": [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else None}
            out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default="posting.json")
    a = ap.parse_args()
    res = Path(a.results)
    rows = load(res, [Path(e) for e in a.extra])
    fams = [("Qwen3-8B", "qwen3-8b", fixed(), 2), ("Qwen3-8B, 1 answer each", "qwen3-8b", fixed(finv=4), 4),
            ("Qwen3-8B, 8 questions", "qwen3-8b", fixed(q=8), 2),
            ("Qwen3-8B, 8 questions, 2 answers each", "qwen3-8b", fixed(q=8, finv=4), 4),
            ("Qwen3-8B, 8 questions, 1 answer each", "qwen3-8b", fixed(q=8, finv=8), 8),
            ("Gemma 3 27B", "gemma-3-27b", fixed(), 2),
            ("Qwen3-4B", "qwen3-4b", fixed(), 2), ("Qwen3-14B", "qwen3-14b", fixed(), 2),
            ("Qwen3-32B", "qwen3-32b", fixed(), 2)]
    out = {"cells": [], "rounds": []}
    pct = lambda v: "   –" if v is None else f"{100 * v:5.1f}"
    for lab, m, cond, bn in fams:
        cs = cells(rows, m, cond)
        base_rows = next((g for (s, n), g in cs.items() if n == bn and s == "results"), None) or \
            next((g for (s, n), g in cs.items() if n == bn), None)
        if not base_rows:
            continue
        base = Base(base_rows)
        for (src, n), g in cs.items():
            st = cell_stats(g)
            tch = target_chances(g)
            st["law_base"] = predict(base, tch)["cr"]
            st["law_cell"] = predict(Base(g), tch)["cr"] if n > bn else st["law_base"]
            out["cells"].append({"family": lab, "source": src, "N": n, **st})
            print(f"{lab:24s} N={n:<4d}{'' if src == 'results' else '(' + src + ')':14s} posted {pct(st['posted'])}  "
                  f"wrong notes {pct(st['wrong_notes'])}  found {pct(st['found'])}  beside wrong {pct(st['beside_wrong'])}  "
                  f"uptake {pct(st['uptake'])} (clean {pct(st['uptake_clean'])}, beside wrong {pct(st['uptake_beside_wrong'])})  "
                  f"CR {pct(st['cr'])}  law(base) {pct(st['law_base'])}  law(cell) {pct(st['law_cell'])}", flush=True)
    # titles in the question's round: holders' writes and seekers' reads, 2 answers each (pairs) against 1 (4 copies)
    from predict_swarm import traces
    out["titles"] = []
    for lab, cond, n in (("2 answers each", fixed(), 2), ("1 answer each", fixed(finv=4), 4),
                         ("8 questions, 4 answers each", fixed(q=8), 2),
                         ("8 questions, 2 answers each", fixed(q=8, finv=4), 4),
                         ("8 questions, 1 answer each", fixed(q=8, finv=8), 8),
                         # the notes list the held answers plus fillers (2 by default): scripts/run_fillers_grid.sh
                         ("2 answers + 1 filler (3 values)", fixed(fillers=1), 2),
                         ("2 answers, no filler (2 values)", fixed(fillers=0), 2),
                         ("1 answer + 3 fillers (4 values)", fixed(finv=4, fillers=3), 4),
                         ("1 answer + 1 filler (2 values)", fixed(finv=4, fillers=1), 4)):
        g = cells(rows, "qwen3-8b", cond).get(("results", n), [])
        if not g:
            continue
        hw, sr = [], []
        for r in g:
            for holds, steps in traces(r):
                for q in range(r["n_questions"]):
                    gen = lambda t: t and t.replace(f"{{Q{q + 1}}}", "{state}")
                    (hw if q in holds else sr).append(gen(steps[q][0] if q in holds else steps[q][2]))
        fav = Counter(t for t in hw if t).most_common(1)[0][0]
        x = {"cell": lab, "N": n, "favourite": fav, "holders_write_favourite": hw.count(fav) / len(hw),
             "seekers_read_favourite": sr.count(fav) / len(sr), "seekers_read_any": sum(map(bool, sr)) / len(sr)}
        out["titles"].append(x)
        print(f"Qwen3-8B, {lab} (N={n}): favourite title {fav!r}: holders post there {pct(x['holders_write_favourite'])}%, "
              f"seekers read there {pct(x['seekers_read_favourite'])}%, seekers read any page {pct(x['seekers_read_any'])}%")
    # per question round, Qwen3-8B with 2 answers each
    cs = cells(rows, "qwen3-8b", fixed())
    base = Base(cs[("results", 2)])
    for (src, n), g in cs.items():
        if n < 4:
            continue
        own, tch = Base(g), target_chances(g)
        pb, po = base.pi(), own.pi()
        seen, cnt = defaultdict(int), defaultdict(int)
        for r in g:
            for ag in r["agents"].values():
                for x in ag["answers"]:
                    if x["chance"]:
                        cnt[x["round"]] += 1
                        seen[x["round"]] += x["seen"]
        for q in range(4):
            sub = [t for t in tch if t[1] == q]
            out["rounds"].append({"source": src, "N": n, "round": q + 1, "found": seen[q + 1] / cnt[q + 1],
                                  "law_base": sum(found(pb, s, qq, dict(hm)) for s, qq, hm in sub) / len(sub),
                                  "law_cell": sum(found(po, s, qq, dict(hm)) for s, qq, hm in sub) / len(sub)})
        rr = [x for x in out["rounds"] if x["source"] == src and x["N"] == n]
        print(f"found by round, Qwen3-8B N={n} {'' if src == 'results' else '(' + src + ')'}: " + " | ".join(
            f"round {x['round']}: {pct(x['found'])} (pairs {pct(x['law_base'])}, cell {pct(x['law_cell'])})" for x in rr))
    # without the rule, paired with the rule-on runs of the same quizzes
    out["rule_off"] = rule_off(rows)
    for x in out["rule_off"]:
        print(f"rule off minus on, {x['model']} N={x['N']} ({x['paired']} paired quizzes): " + " | ".join(
            f"{k} {pct(x[k]['on'])} -> {pct(x[k]['off'])} [{100 * x[k]['ci'][0]:+.1f}, {100 * x[k]['ci'][1]:+.1f}]"
            for k in RULE_KEYS if x[k]["ci"]))
    (res / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {res / a.out}")


if __name__ == "__main__":
    main()
