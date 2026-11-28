#!/usr/bin/env python3
"""Pages for gpt-oss-120b as a reader in the grid test (results/predictions_grid.md): Gemma's answer page from item 26
rewritten by Gemma, Qwen3-8B and gpt-oss-120b (results/paraphrase/gemma_by_<writer>.json), each moved to the title
gpt-oss-120b's late holders read in round 1 in pairs: the state's name alone ("Iowa"; 68% of their round-1 reads in
the or_pairs runs, results/swarm_or_pairs). The notes are unchanged.

  .venv/bin/python scripts/make_reader_pages.py        writes results/reader/gptoss_by_<writer>.json
"""
from __future__ import annotations

import json
from pathlib import Path

SC = Path(__file__).resolve().parent.parent
WRITERS = ("gemma", "q8", "gptoss")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--para", default=str(SC / "results" / "paraphrase"))
    ap.add_argument("--out", default=str(SC / "results" / "reader"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for w in WRITERS:
        d = json.loads((Path(a.para) / f"gemma_by_{w}.json").read_text())
        pages = {s: {"title": p["title"].rsplit(" Construction", 1)[0], "notes": p["notes"]} for s, p in d["pages"].items()}
        (out / f"gptoss_by_{w}.json").write_text(json.dumps(
            {"source": f"results/paraphrase/gemma_by_{w}.json", "writer": d["writer"], "title": "the state's name",
             "pages": pages}, indent=1, ensure_ascii=False) + "\n")
        print(f"gptoss_by_{w}: {len(pages)} pages, e.g. {pages['10009']['title']!r}, {len(pages['10009']['notes'])} notes")


if __name__ == "__main__":
    main()
