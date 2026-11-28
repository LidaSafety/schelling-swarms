#!/usr/bin/env python3
"""Manipulation check for the own-style test (results/predictions_paraphrase.md): can each model tell its own notes
from the other model's? With both pods up (`eval "$(python3 scripts/pod.py endpoints)"`):

  .venv/bin/python scripts/check_authorship.py      writes results/paraphrase_authorship.json
  .venv/bin/python scripts/check_authorship.py --by gemma q8 llama --out results/writers_authorship.json
      the same for the writers test (results/predictions_writers.md): its own notes and each writer's rewrite
  .venv/bin/python scripts/check_authorship.py --by gemma q8 gptoss --with-gptoss --out results/grid_authorship.json
      the grid test (results/predictions_grid.md): also gpt-oss-120b as a judge, through OpenRouter (export the
      OpenRouter address and key first), of the three rewrites of Gemma's page that it reads (results/reader/)

For each reader (Gemma 3 27B, Qwen3-8B), 150 notes drawn at random (seeded) from each of the three pages it reads in
the test (its own game's answers, rewritten by itself, rewritten by the other model), each shown alone with ASK
(temperature 0, 10 tokens). Reported: the share answered "me", per page. Reported, not registered.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SC = Path(__file__).resolve().parent.parent
MODELS = {"gemma": "gemma-3-27b", "q8": "qwen3-8b"}
ASK = ("Several AI assistants write notes on a shared wiki. Some notes there were written by you (another session of "
       "the same model as you), and some by a different AI model.\n\nNote: \"{note}\"\n\nWho wrote this note? Answer "
       "with one word: me or other.")
N = 150


def pages_for(reader: str, pages: Path, para: Path, by: list[str] | None = None) -> dict[str, list[str]]:
    other = next(m for m in MODELS if m != reader)
    files = {"own": pages / f"{reader}_answers_q1.json", "self": para / f"{reader}_by_{reader}.json",
             "other": para / f"{reader}_by_{other}.json"}
    if by:
        files = {"own": pages / f"{reader}_answers_q1.json", **{f"by_{w}": para / f"{reader}_by_{w}.json" for w in by}}
    out = {}
    for k, f in files.items():
        d = json.loads(f.read_text())
        bad = set(d.get("failed", []))
        out[k] = [t for s, p in sorted(d["pages"].items()) if s not in bad for t in p["notes"]]
    return out


def judge(llm, note: str, seed: int) -> str:
    t = llm.chat([{"role": "user", "content": ASK.format(note=note)}], temperature=0.0, max_tokens=10,
                 seed=seed)["text"].strip().lower()
    return "me" if t.startswith("me") else "other" if t.startswith("other") else "unclear"


def check(llms: dict, pages: Path, para: Path, by: list[str] | None = None, reader_pages: Path | None = None) -> dict:
    out = {}
    for reader, llm in llms.items():
        if reader == "gptoss":     # reads Gemma's page, rewritten by each writer and retitled (no page of its own)
            notes = {f"by_{w}": [t for _, p in sorted(json.loads((reader_pages / f"gptoss_by_{w}.json").read_text())
                                                      ["pages"].items()) for t in p["notes"]] for w in by}
        else:
            notes = pages_for(reader, pages, para, by)
        out[reader] = {}
        for k, pool in notes.items():
            sample = random.Random(f"authorship-{reader}-{k}").sample(pool, min(N, len(pool)))
            with ThreadPoolExecutor(16) as ex:
                ans = list(ex.map(lambda x: judge(llm, x[1], x[0]), enumerate(sample)))
            out[reader][k] = {"n": len(ans), "me": ans.count("me") / max(1, len(ans)), "unclear": ans.count("unclear")}
            print(f"{reader} reading {k:8s}: {100 * out[reader][k]['me']:.1f}% answered 'me' "
                  f"({out[reader][k]['unclear']} unclear of {len(ans)})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default=str(SC / "results" / "page_content"))
    ap.add_argument("--para", default=str(SC / "results" / "paraphrase"))
    ap.add_argument("--out", default=str(SC / "results" / "paraphrase_authorship.json"))
    ap.add_argument("--by", nargs="+", default=None, help="writers whose rewrites to show (default: self and other)")
    ap.add_argument("--with-gptoss", action="store_true", help="also gpt-oss-120b as a judge (the grid test)")
    ap.add_argument("--reader-pages", default=str(SC / "results" / "reader"))
    a = ap.parse_args()
    import os
    from scb.cli import _llm
    from scb.llm import LLM
    llms = {k: _llm(m, 0.0, 10) for k, m in MODELS.items()}
    if a.with_gptoss:          # its reasoning needs room: 40x the 10-token budget
        llms["gptoss"] = LLM(model="openai/gpt-oss-120b", base_url=os.environ["SB_BASE_URL"],
                             api_key=os.environ["SB_API_KEY"], temperature=0.0, max_tokens=10, token_mult=40.0,
                             extra_body={"provider": {"order": ["Crusoe"], "allow_fallbacks": False,
                                                      "require_parameters": True}, "reasoning": {"effort": "low"}})
    out = check(llms, Path(a.pages), Path(a.para), a.by, Path(a.reader_pages))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
