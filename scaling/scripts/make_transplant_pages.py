#!/usr/bin/env python3
"""Pages to plant for the transplant test (results/predictions_transplant.md): real round-1 pages of Gemma 3 27B
swarms, from the aaai branch's 32- and 64-copy runs, written as files the runner plants before round 1 of a pair.

  .venv/bin/python scripts/make_transplant_pages.py aaai
      reads aaai/swarm_gemma_ladder_n32 and aaai/swarm_gemma_ladder_n64 (the aaai branch's result folders, checked out
      into a folder of their own) and writes results/transplant/{page64,holders64,seekers64,page32}.json

A game's round-1 page is every note written in round 1 under the title most of its copies wrote under, in the order
the page showed them. Holders' notes are those of the copies holding question 1 (the early holders), seekers' notes
those of the other copies (the late holders, which only search in round 1). The last holder's note and the last
seeker's note are left out of every page, because the pair's own two copies, one of each, add theirs: a late holder
that reads the planted title in round 1 then sees the swarm's count and mix of notes. The arms:
  page64     the 64-copy game's page
  holders64  its holders' notes only
  seekers64  its seekers' notes only
  page32     the 32-copy game's page
Each file maps a seed to {"title", "notes"}. The title is the pair's usual round-1 title for question 1,
"<State> Construction Employment Share" (PLANT_STYLES in the runner), whatever the source page's title; 95 of the 100
64-copy games and 96 of the 32-copy games already use it. Every page is checked against the page text a late holder
of that game was shown when it read the page in round 1.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

SC = Path(__file__).resolve().parent.parent
SOURCES = {64: "swarm_gemma_ladder_n64", 32: "swarm_gemma_ladder_n32"}
TITLE = "{state} Construction Employment Share"


def round1_text(r: dict, a: str) -> str:
    """The text of copy a's round-1 note, from its write action (1 system message, then 6 per round)."""
    edit = json.loads(r["agents"][a]["messages"][2]["content"]).get("edit")
    return str(edit.get("text", "")) if isinstance(edit, dict) else ""


def page(r: dict) -> tuple[str, list[tuple[bool, str]]]:
    """The round-1 page under the title most copies wrote under: (title, [(is a holder's note, text)]) in seat order,
    checked against what a late holder that read it was shown."""
    ev, q1 = r["events"], r["questions"][0]
    top = Counter(w["path"] for a in ev for w in ev[a]["writes"] if w["round"] == 1 and w.get("path")).most_common(1)[0][0]
    notes = [(q1 in r["holds"][a], round1_text(r, a)) for a in ev
             if any(w["round"] == 1 and w.get("path") == top for w in ev[a]["writes"])]
    reader = next(a for a in ev if q1 not in r["holds"][a]
                  and any(x["round"] == 1 and x.get("path") == top for x in ev[a]["reads"]))
    shown = r["agents"][reader]["messages"][5]["content"]
    body = "\n\n".join(t for _, t in notes)
    assert body in shown, f"seed {r['seed']}: the rebuilt page differs from what copy {reader} was shown"
    return top, notes


def arms(rows: list[dict], n: int) -> dict[str, dict]:
    """The arms built from one swarm size: seed -> {"title", "notes"}, and the counts behind them."""
    out = {f"page{n}": {}, f"holders{n}": {}, f"seekers{n}": {}} if n == 64 else {f"page{n}": {}}
    for r in sorted(rows, key=lambda r: r["seed"]):
        _, notes = page(r)
        drop = {max(i for i, (h, _) in enumerate(notes) if h), max(i for i, (h, _) in enumerate(notes) if not h)}
        kept = [(h, t) for i, (h, t) in enumerate(notes) if i not in drop]
        title = TITLE.format(state=r["questions"][0])
        pick = {f"page{n}": lambda h: True, f"holders{n}": lambda h: h, f"seekers{n}": lambda h: not h}
        for arm in out:
            out[arm][str(r["seed"])] = {"title": title, "notes": [t for h, t in kept if pick[arm](h)],
                                        "holders": sum(h for h, _ in kept if pick[arm](h)),
                                        "seekers": sum(not h for h, _ in kept if pick[arm](h))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="folder holding the aaai branch's swarm_gemma_ladder_n32 and _n64")
    ap.add_argument("--out", default=str(SC / "results" / "transplant"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for n, tag in SOURCES.items():
        f = Path(a.source) / tag / "raw" / "gemma-3-27b.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(f, "rt")]
        assert sorted(r["seed"] for r in rows) == list(range(10000, 10100)), f"{f}: expected seeds 10000-10099"
        assert all(r["awareness"] == "unaware" and r["rule"] == "forbidden" and r["n_agents"] == n for r in rows)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        for arm, pages in arms(rows, n).items():
            k = [len(p["notes"]) for p in pages.values()]
            hs = [p["holders"] for p in pages.values()]
            ss = [p["seekers"] for p in pages.values()]
            (out / f"{arm}.json").write_text(json.dumps(
                {"arm": arm, "source": f"aaai branch, scaling/results/{tag}/raw/gemma-3-27b.jsonl.gz", "sha256": sha,
                 "pages": {s: {"title": p["title"], "notes": p["notes"]} for s, p in pages.items()}},
                indent=1, ensure_ascii=False) + "\n")
            print(f"{arm:10s} {len(pages)} pages; planted notes per page median {np.median(k):.0f} (range {min(k)}-"
                  f"{max(k)}): holders' {np.median(hs):.0f}, seekers' {np.median(ss):.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
