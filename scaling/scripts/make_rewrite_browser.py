#!/usr/bin/env python3
"""The rewrite browser (docs/planted_page_rewrites.html): every planted answer page of the own-style, writers and grid
tests (items 28-30), quiz by quiz, as written and as rewritten by each writer, with how often late holders posted
their own answer in round 2 after reading each version.

  .venv/bin/python scripts/make_rewrite_browser.py
      reads results/page_content/, results/paraphrase/, results/grid.json, results/writers.json and the game logs of
      the plant_grid and plant_writer tags; fills docs/planted_page_rewrites.template.html and writes
      docs/planted_page_rewrites.html, a single self-contained page

Per quiz and version it counts the late holders who read the planted page in round 1 (as in
scripts/analyze_page_content.py) and how many posted their round-2 answer, and keeps one example of the note a late
holder wrote in round 2. Llama's version was read in the writers runs (item 29); the rest in the grid runs (item 30).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

SC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SC / "scripts"))
from analyze_page_content import late_record  # noqa: E402
from analyze_transplant import rows_of  # noqa: E402
from title_law import condition, fixed  # noqa: E402

VERS = ("orig", "gemma", "q8", "llama", "gptoss")
data = {"sources": {}, "outcomes": {}, "quiz": {}}
for src in ("gemma", "q8"):
    files = {"orig": SC / "results" / "page_content" / f"{src}_answers_q1.json",
             **{w: SC / "results" / "paraphrase" / f"{src}_by_{w}.json" for w in VERS[1:]}}
    pages = {v: json.loads(f.read_text())["pages"] for v, f in files.items()}
    data["sources"][src] = {s: {"title": pages["orig"][s]["title"], **{v: pages[v][s]["notes"] for v in VERS}}
                            for s in pages["orig"]}

def note2(r, name):
    raw = r["agents"][name]["messages"][8]["content"]
    try:
        e = json.loads(raw).get("edit")
        return e.get("text", "") if isinstance(e, dict) else "(wrote nothing)"
    except (ValueError, AttributeError):
        return raw[:300]

def outcomes(tag, model, usual, reader_prefix):
    out = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))
    for r in rows_of(SC / "results" / f"swarm_{tag}"):
        if r["model"] != model or condition(r) != fixed():
            continue
        p = r.get("plant")
        f = p["file"] if isinstance(p, dict) and "file" in p else None
        arm = "nothing" if f is None else f.removeprefix(f"{reader_prefix}_by_")
        x = late_record(r, p["title"] if isinstance(p, dict) else usual.format(state=r["questions"][0]))
        if not x or not x["treated"]:
            continue
        qs = r["questions"]
        late = next(n for n in r["holds"] if qs[0] not in r["holds"][n] and qs[1] in r["holds"][n])
        c = out[arm][str(r["seed"])]
        c[0] += x["post2"]; c[1] += 1
        if c[2] is None:
            c[2] = note2(r, late)
        data["quiz"].setdefault(str(r["seed"]), {"q": qs})
    return out

readers = {"gemma": ("plant_grid_gemma", "gemma-3-27b", "{state} Construction Employment Share", "gemma", "gemma"),
           "q8": ("plant_grid_q8", "qwen3-8b", "{state} construction occupation share", "q8", "q8"),
           "gptoss": ("plant_grid_gptoss", "openai/gpt-oss-120b", "{state}", "gptoss", "gemma")}
for reader, (tag, model, usual, prefix, src) in readers.items():
    o = outcomes(tag, model, usual, prefix)
    if reader in ("gemma", "q8"):        # Llama's version was read in the writers runs (item 29)
        wtag = f"plant_writer_{reader}"
        ow = outcomes(wtag, model, usual, prefix)
        o["llama"] = ow["llama"]
    data["outcomes"][reader] = {"source": src, "arms": {a: dict(v) for a, v in o.items()}}
grid = json.loads((SC / "results" / "grid.json").read_text())
writers = json.loads((SC / "results" / "writers.json").read_text())
data["summary"] = {r: {**{a: c["post2"] for a, c in grid["cells"][r].items()}} for r in grid["cells"]}
for r, m in (("gemma", "gemma-3-27b"), ("q8", "qwen3-8b")):
    data["summary"][r]["llama"] = writers["models"][m]["arms"]["by_llama"]["post2"]


def main() -> None:
    template = (SC / "docs" / "planted_page_rewrites.template.html").read_text()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    out = SC / "docs" / "planted_page_rewrites.html"
    out.write_text(template.replace("__DATA__", blob))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
