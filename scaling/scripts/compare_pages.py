#!/usr/bin/env python3
"""Print the wiki text of the own-style, writers and grid tests (items 28-30) side by side.

  .venv/bin/python scripts/compare_pages.py notes gemma 10009 [--n 5]
      the first N notes of one answer page (source: gemma or q8, i.e. whose 64-copy game it comes from), as written
      and as rewritten by each writer (results/page_content/, results/paraphrase/)
  .venv/bin/python scripts/compare_pages.py games plant_grid_gemma 10009 [--offset 6]
      for one quiz in one run: per arm, the start of the page the late holder read in round 1 and the note it wrote in
      round 2 (from the game logs, results/swarm_<tag>/)
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import textwrap
from pathlib import Path

SC = Path(__file__).resolve().parent.parent
WRITERS = {"gemma": "Gemma 3 27B", "q8": "Qwen3-8B", "llama": "Llama 3.3 70B", "gptoss": "gpt-oss-120b"}


def wrap(label: str, text: str, width: int = 100) -> str:
    return textwrap.fill(text, width, initial_indent=f"  {label:18s}", subsequent_indent=" " * 20)


def show_notes(source: str, seed: str, n: int) -> None:
    versions = {"as written": SC / "results" / "page_content" / f"{source}_answers_q1.json"}
    versions |= {f"by {name}": SC / "results" / "paraphrase" / f"{source}_by_{w}.json" for w, name in WRITERS.items()}
    pages = {k: json.loads(f.read_text())["pages"][seed] for k, f in versions.items() if f.is_file()}
    counts = {k: len(p["notes"]) for k, p in pages.items()}
    title = pages["as written"]["title"]
    print(f"{source}'s answer page for seed {seed}, title {title!r}; notes per version: {counts}")
    if len(set(counts.values())) > 1:
        print("  (a version left a note off, so notes after it are not aligned)")
    for i in range(min(n, *counts.values())):
        print(f"\nnote {i + 1}")
        for k, p in pages.items():
            print(wrap(k, f"{p['notes'][i]}   [{len(p['notes'][i])} chars]"))


def note_text(r: dict, name: str, rnd: int) -> str:
    raw = r["agents"][name]["messages"][2 + 6 * (rnd - 1)]["content"]
    try:
        return str(json.loads(raw).get("edit") or "(wrote nothing)")
    except ValueError:
        return raw


def show_games(tag: str, seed: int, offset: int | None) -> None:
    f = sorted((SC / "results" / f"swarm_{tag}" / "raw").glob("*.jsonl*"))[0]
    rows = [json.loads(x) for x in (gzip.open(f, "rt") if f.suffix == ".gz" else open(f))]
    rows = [r for r in rows if "error" not in r and r["seed"] == seed]
    offsets = sorted({r.get("sample_offset", 0) for r in rows})
    off = offsets[0] if offset is None else offset
    rows = [r for r in rows if r.get("sample_offset", 0) == off]
    print(f"{tag}, seed {seed}, sample offset {off} (offsets in this run: {offsets})")
    for r in sorted(rows, key=lambda r: (r.get("plant") or {}).get("file", "")):
        arm = (r.get("plant") or {}).get("file", "nothing planted")
        qs = r["questions"]
        late = next(n for n in r["holds"] if qs[0] not in r["holds"][n] and qs[1] in r["holds"][n])
        rd = next((x for x in r["events"][late]["reads"] if x["round"] == 1), None)
        shown = r["agents"][late]["messages"][5]["content"]
        body = shown.split("\n", 1)[1] if "\n" in shown else shown
        first = [x for x in body.split("\n\n") if x.strip() and "ANSWER step" not in x][:2]
        posted = any(w["round"] == 2 and qs[1] in w["posts"] for w in r["events"][late]["writes"])
        extra = f" ({rd['planted']} planted notes)" if rd and "planted" in rd else ""
        print(f"\n[{arm}]  round 1 the late holder read {rd['path'] if rd else None!r}{extra}")
        for x in first:
            print(wrap("page starts:", re.sub(r"\s+", " ", x)[:300]))
        print(wrap(f"round 2 ({qs[1]}):", f"{note_text(r, late, 2)}   -> {'posted its answer' if posted else 'did not post'}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("notes")
    a1.add_argument("source", choices=["gemma", "q8"])
    a1.add_argument("seed")
    a1.add_argument("--n", type=int, default=5)
    a2 = sub.add_parser("games")
    a2.add_argument("tag")
    a2.add_argument("seed", type=int)
    a2.add_argument("--offset", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "notes":
        show_notes(a.source, a.seed, a.n)
    else:
        show_games(a.tag, a.seed, a.offset)


if __name__ == "__main__":
    main()
