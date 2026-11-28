#!/usr/bin/env python3
"""Exploratory checks on the rewrite tests (items 28-30), not registered.

1. Belief: of the late holders who read the planted title in round 1, the share that saw the planted value and the
   share of those that answered question 1 with it, per run, reader and writer, beside their round-2 posting. If some
   writers' notes were believed more, readers would take the value from them more often.
2. Voice: the share of each rewritten page's notes in the first person ("My research indicates ..."), per writer, and
   whether pages with more first-person notes give more posting: within quiz (across writers), and within quiz and
   writer, with 95% intervals from 2,000 resamples of the quizzes.

  .venv/bin/python scripts/analyze_rewrites_extra.py      # writes results/rewrites_extra.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_page_content import late_record  # noqa: E402
from analyze_transplant import rows_of  # noqa: E402
from title_law import condition, fixed  # noqa: E402

SC = Path(__file__).resolve().parent.parent
READERS = {  # reader: model, usual round-1 title, whose answer page it reads, runs
    "gemma": ("gemma-3-27b", "{state} Construction Employment Share", "gemma", ("plant_grid_gemma", "plant_writer_gemma")),
    "q8": ("qwen3-8b", "{state} construction occupation share", "q8", ("plant_grid_q8", "plant_writer_q8")),
    "gptoss": ("openai/gpt-oss-120b", "{state}", "gemma", ("plant_grid_gptoss",)),
}
WRITERS = ("gemma", "q8", "gptoss", "llama")
FIRST = re.compile(r"\b(I|I'm|I've|I'll|I'd|my|me|mine|our|we)\b", re.I)
REPS = 2000


def records(reader: str) -> dict[str, list[dict]]:
    """Per run: the late holders that read the planted title (nothing planted: the same title)."""
    model, usual, _, tags = READERS[reader]
    out = {}
    for tag in tags:
        recs = []
        for r in rows_of(SC / "results" / f"swarm_{tag}"):
            if r["model"] != model or r["n_agents"] != 2 or condition(r) != fixed():
                continue
            p = r.get("plant")
            f = p["file"] if isinstance(p, dict) and "file" in p else None
            x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
            if not x or not x["treated"]:
                continue
            qs = r["questions"]
            late = next(a for a in r["holds"] if qs[0] not in r["holds"][a] and qs[1] in r["holds"][a])
            recs.append({"writer": "nothing" if f is None else f.split("_by_")[-1], "seed": str(r["seed"]),
                         "saw": bool(r["agents"][late]["answers"][0].get("seen_planted")),
                         "took": bool(x["took_plant"]), "post2": bool(x["post2"])})
        out[tag] = recs
    return out


def belief(recs: list[dict]) -> dict:
    out = {}
    for w in ("nothing",) + WRITERS:
        R = [x for x in recs if x["writer"] == w]
        if not R:
            continue
        saw = [x for x in R if x["saw"]]
        out[w] = {"late_holders": len(R), "saw_planted": sum(x["saw"] for x in R) / len(R),
                  "took_value": sum(x["took"] for x in saw) / len(saw) if saw else None,
                  "post2": sum(x["post2"] for x in R) / len(R)}
    return out


def first_person(src: str) -> dict[str, dict[str, float]]:
    """Per writer and quiz: the share of the rewritten page's notes in the first person."""
    out = {}
    for w in WRITERS:
        f = SC / "results" / "paraphrase" / f"{src}_by_{w}.json"
        if f.is_file():
            pages = json.loads(f.read_text())["pages"]
            out[w] = {s: float(np.mean([bool(FIRST.search(n)) for n in p["notes"]])) for s, p in pages.items()
                      if p["notes"]}
    return out


def slopes(recs: list[dict], fp: dict[str, dict[str, float]], rng: np.random.Generator) -> dict:
    """Posting points per 10 points more first-person notes, by weighted least squares on quiz-by-writer cells."""
    cells = defaultdict(lambda: [0, 0])
    for x in recs:
        if x["writer"] in fp and x["seed"] in fp[x["writer"]]:
            c = cells[(x["seed"], x["writer"])]
            c[0] += x["post2"]
            c[1] += 1
    by_seed = defaultdict(list)
    for (s, w), (k, n) in cells.items():
        by_seed[s].append((WRITERS.index(w), fp[w][s], k / n, n))
    seeds = sorted(by_seed)

    def fit(draw: list[str], twoway: bool) -> float:
        rows = [(i, w, x, y, n) for i, s in enumerate(draw) for w, x, y, n in by_seed[s]]
        g, wr, x, y, n = (np.array(v) for v in zip(*rows))
        n = n.astype(float)

        def demean(v, grp):
            return v - (np.bincount(grp, n * v) / np.bincount(grp, n))[grp]

        for _ in range(30 if twoway else 1):
            x, y = demean(x, g), demean(y, g)
            if twoway:
                x, y = demean(x, wr), demean(y, wr)
        sxx = float(np.sum(n * x * x))
        return 10 * float(np.sum(n * x * y)) / sxx if sxx > 0 else float("nan")

    out = {}
    for name, twoway in (("within_quiz", False), ("within_quiz_and_writer", True)):
        boots = [fit(list(rng.choice(seeds, len(seeds))), twoway) for _ in range(REPS)]
        out[name] = {"value": fit(seeds, twoway), "ci": [float(v) for v in np.nanpercentile(boots, [2.5, 97.5])]}
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    res = {"belief": {}, "first_person_pages": {}, "first_person_slopes": {}}
    for src in ("gemma", "q8"):
        fp = first_person(src)
        res["first_person_pages"][src] = {w: float(np.mean(list(v.values()))) for w, v in fp.items()}
    for reader, (_, _, src, _) in READERS.items():
        recs = records(reader)
        res["belief"][reader] = {tag: belief(r) for tag, r in recs.items()}
        pooled = [x for r in recs.values() for x in r]
        if any(res["first_person_pages"][src].values()):
            res["first_person_slopes"][reader] = slopes(pooled, first_person(src), rng)
    (SC / "results" / "rewrites_extra.json").write_text(json.dumps(res, indent=1))
    for reader, runs in res["belief"].items():
        for tag, cells in runs.items():
            print(f"{reader:7s} {tag:20s} " + "  ".join(
                f"{w} post {100 * c['post2']:.1f}" + (f" took {100 * c['took_value']:.1f}" if c["took_value"] is not None else "")
                for w, c in cells.items()))
    print("first-person share of notes:", {s: {w: round(100 * v) for w, v in d.items()}
                                           for s, d in res["first_person_pages"].items()})
    for reader, d in res["first_person_slopes"].items():
        print(reader, {k: (round(v["value"], 1), [round(c, 1) for c in v["ci"]]) for k, v in d.items()})


if __name__ == "__main__":
    main()
