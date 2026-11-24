#!/usr/bin/env python3
"""Pages to plant for the page-content test (results/predictions_page_content.md): what on a crowded page makes a late
holder post. Each page is built from the round-1 page of the model's own 64-copy game with the same quiz.

  .venv/bin/python scripts/make_content_pages.py aaai
      reads aaai/swarm_gemma_ladder_n64 (the aaai branch's result folder, checked out as for make_transplant_pages.py)
      and ../wiki_game/data/swarm4round/swarm_ref_8b_n64, and writes results/page_content/<model>_<arm>.json

A game's round-1 page is every note written in round 1 under the title most of its copies wrote under. Its answers
are the notes of copies holding question 1 that carry its value; its requests are the notes of the other copies that
carry no value. Each arm plants as many notes as the page has of that kind, less one, because the pair's own copy of
that kind adds its own. Answer notes keep the game's wording; a note that does not name the state (a bare number, as
most of Qwen3-8B's are) gets "<State>: " in front, in every answer arm, so the answer arms differ only in the state and
the value they give. Requests use each model's most common request. The arms:
  answers_q1        answers to question 1
  answers_unasked   the same notes about a state outside the quiz and outside both copies' notes, whose value is
                    none of the quiz's four
  answers_q3        the same notes about question 3's state (Gemma only)
  answers_q1_other  answers to question 1 in Qwen3-8B's most common form, a bare number, named as in the other arms
                    ("<State>: <value>"; Gemma only: Gemma's own notes mostly paraphrase the question in a sentence)
  requests_q1       requests for question 1
  requests_q2       requests for question 2, the late holder's own question
  answers_q2        answers to question 2: the late holder's own answer, already posted
Values are the quiz's own: the runner draws one for every state (scb/swarm.py), and this script repeats its draws,
checked against the asked questions of each source game. Each file maps a seed to {"title", "notes"}; the title is the
pair's usual round-1 title for question 1. In a few seeds question 2's or question 3's value equals question 1's, so
their notes also carry question 1's value on its page; the script lists those seeds, and scripts/analyze_page_content.py
leaves them out of those arms.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from scb.quiz import STATES

SC = Path(__file__).resolve().parent.parent

N_FILLERS = 2
TIES = ("answers_q3", "answers_q2")     # arms whose state's value can equal question 1's
MODELS = {
    "gemma": {"source": "swarm_gemma_ladder_n64/raw/gemma-3-27b.jsonl.gz", "title": "{state} Construction Employment Share",
              "request": "Unknown. Need to find data for {state}.",
              "arms": ("answers_q1", "answers_unasked", "answers_q3", "answers_q1_other", "requests_q1", "requests_q2",
                       "answers_q2")},
    "q8": {"source": "swarm_ref_8b_n64", "title": "{state} construction occupation share",
           "request": "The share of employed workers in construction occupations in {state} is not known.",
           "arms": ("answers_q1", "answers_unasked", "requests_q1", "requests_q2", "answers_q2")},
}


def quiz(seed: int, q: int = 4) -> tuple[list[str], dict[str, float], set[str]]:
    """The questions, every state's value, and the states in a pair's notes, drawn as scb/swarm.py draws them."""
    rng = random.Random(seed)
    qs = rng.sample(STATES, q)
    truth = {s: round(rng.uniform(4.0, 12.0), 1) for s in STATES}
    fillers = [s for s in STATES if s not in qs]
    in_notes = {s for _ in range(2) for s in rng.sample(fillers, N_FILLERS)}
    return qs, truth, in_notes


def main_page(r: dict) -> tuple[list[str], list[str]]:
    """The answers and the requests on a game's main round-1 page, in the order the page showed them."""
    qs, ev, holds = r["questions"], r["events"], r["holds"]
    w1 = {a: next((w for w in ev[a]["writes"] if w["round"] == 1), None) for a in ev}
    top = Counter(w["path"] for w in w1.values() if w and w.get("path")).most_common(1)[0][0]
    answers, requests = [], []
    for a, w in w1.items():
        if not w or w.get("path") != top:
            continue
        text = str(json.loads(r["agents"][a]["messages"][2]["content"]).get("edit", {}).get("text", ""))
        if qs[0] in holds[a] and qs[0] in w["posts"]:
            answers.append(text)
        elif qs[0] not in holds[a] and not w["posts"]:
            requests.append(text)
    return answers, requests


def swap(text: str, q1: str, v1: float, state: str, value: float) -> str:
    """An answer note in named form, giving `state` and `value` in place of question 1's."""
    if not re.search(rf"\b{re.escape(q1)}\b", text):
        text = f"{q1}: {text}"
    old = f"{v1:.1f}"
    assert re.search(rf"(?<![\d.]){re.escape(old)}(?!\d)", text), (text, old)
    text = re.sub(rf"(?<![\d.]){re.escape(old)}(?!\d)", f"{value:.1f}", text)
    return re.sub(rf"\b{re.escape(q1)}\b", state, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aaai", help="folder holding the aaai branch's swarm_gemma_ladder_n64")
    ap.add_argument("--swarm4round", default=str(SC.parent / "wiki_game" / "data" / "swarm4round"))
    ap.add_argument("--out", default=str(SC / "results" / "page_content"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, m in MODELS.items():
        if name == "gemma":
            f, where = Path(a.aaai) / m["source"], f"aaai branch, scaling/results/{m['source']}"
        else:
            f = sorted((Path(a.swarm4round) / m["source"] / "raw").glob("*.jsonl*"))[0]
            where = f"wiki_game/data/swarm4round/{m['source']}/raw/{f.name}"
        rows = [json.loads(line) for line in (gzip.open(f, "rt") if f.suffix == ".gz" else open(f))]
        rows = sorted((r for r in rows if "error" not in r), key=lambda r: r["seed"])
        assert [r["seed"] for r in rows] == list(range(10000, 10100)), f"{f}: expected seeds 10000-10099"
        files = {arm: {} for arm in m["arms"]}
        ties = {arm: [] for arm in TIES}
        for r in rows:
            s = r["seed"]
            qs, truth, in_notes = quiz(s)
            assert qs == r["questions"], f"{name} seed {s}: the quiz does not repeat"
            for ag in r["agents"].values():
                for x in ag["answers"]:
                    assert abs(x["truth"] - truth[x["state"]]) < 1e-9, f"{name} seed {s}: values do not repeat"
            quiz_values = {f"{truth[x]:.1f}" for x in qs}
            unasked = random.Random(f"unasked-{s}").choice(sorted(
                x for x in set(STATES) - set(qs) - in_notes if f"{truth[x]:.1f}" not in quiz_values))
            answers, requests = main_page(r)
            answers, requests = answers[:-1], requests[:-1]          # the pair's own two notes take these places
            q1, v1 = qs[0], truth[qs[0]]
            make = {
                "answers_q1": lambda: [swap(t, q1, v1, q1, v1) for t in answers],
                "answers_unasked": lambda: [swap(t, q1, v1, unasked, truth[unasked]) for t in answers],
                "answers_q3": lambda: [swap(t, q1, v1, qs[2], truth[qs[2]]) for t in answers],
                "answers_q1_other": lambda: [f"{q1}: {v1:.1f}" for _ in answers],
                "requests_q1": lambda: [m["request"].format(state=q1) for _ in requests],
                "requests_q2": lambda: [m["request"].format(state=qs[1]) for _ in requests],
                "answers_q2": lambda: [swap(t, q1, v1, qs[1], truth[qs[1]]) for t in answers],
            }
            for arm in m["arms"]:
                files[arm][str(s)] = {"title": m["title"].format(state=q1), "notes": make[arm]()}
                carry = any(f"{v1:.1f}" in t for t in files[arm][str(s)]["notes"])   # as the runner checks a page
                assert carry == (arm in ("answers_q1", "answers_q1_other") and bool(answers)) or arm in TIES, (name, s, arm)
                if carry and arm in TIES:
                    ties[arm].append(s)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        for arm, pages in files.items():
            k = [len(p["notes"]) for p in pages.values()]
            (out / f"{name}_{arm}.json").write_text(json.dumps(
                {"arm": arm, "model": name, "source": where, "sha256": sha, "pages": pages},
                indent=1, ensure_ascii=False) + "\n")
            print(f"{name}_{arm:17s} {len(pages)} pages; notes per page median {np.median(k):.0f} (range {min(k)}-{max(k)})"
                  + (f"; value equals question 1's in seeds {ties[arm]}" if ties.get(arm) else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
