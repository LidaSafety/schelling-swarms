#!/usr/bin/env python3
"""Pages for the own-style test (results/predictions_paraphrase.md): the answer page of each model's 64-copy game
(results/page_content/<model>_answers_q1.json, item 26), rewritten note by note by Gemma 3 27B and by Qwen3-8B.

  scripts/run_on_pod.sh is not needed: with both pods up, `eval "$(python3 scripts/pod.py endpoints)"`, then
  .venv/bin/python scripts/make_paraphrase_pages.py
      writes results/paraphrase/<source>_by_<writer>.json for source and writer in {gemma, q8}
  .venv/bin/python scripts/make_paraphrase_pages.py --writers llama     (after `eval "$(python3 scripts/pod.py
      exports openrouter)"`) adds the third writer of the writers test (results/predictions_writers.md): Llama 3.3 70B
      Instruct through OpenRouter, pinned to one host without fallbacks
  .venv/bin/python scripts/make_paraphrase_pages.py --writers gptoss    likewise adds the third writer of the grid test
      (results/predictions_grid.md): gpt-oss-120b through OpenRouter, pinned to one host, reasoning effort low, with 6x
      the token budget for its reasoning, as in its runs as a player

Each note is sent alone to the writer with REWRITE (temperature 0.7, up to 150 tokens, the model's own serving
settings, thinking off for Qwen3-8B). A rewrite must name the note's state and no other state and carry the note's
value exactly and no other number; it should also be within max(15, 25%) characters of the original's length. A
rewrite that misses either is asked for again, with a new sampling seed, up to 8 times. After 8 tries the rewrite
with the right content that is closest in length is kept ("approx"); a note with no rewrite of the right content is
left off its page ("dropped"). Each file maps a seed to {"title", "notes"}, the title and note order of the source page,
and gives these counts and the mean note length before and after.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scb.quiz import STATES

SC = Path(__file__).resolve().parent.parent
MODELS = {"gemma": "gemma-3-27b", "q8": "qwen3-8b"}
LLAMA = "meta-llama/llama-3.3-70b-instruct"
LLAMA_HOST = "DeepInfra"
GPTOSS = "openai/gpt-oss-120b"
GPTOSS_HOST = "Crusoe"
WRITERS = {**MODELS, "llama": LLAMA, "gptoss": GPTOSS}
REWRITE = ('Here is a note that someone wrote on a shared wiki page titled "{title}":\n\n{note}\n\n'
           "Rewrite this note in your own words, the way you would write it yourself. Keep its meaning, the state's "
           "name and the number {value} exactly, and keep it about the same length (about {n} characters). Reply "
           "with the rewritten note only.")
TRIES = 8
NUM = re.compile(r"\d+(?:\.\d+)?")


def value_of(note: str, title: str) -> tuple[str, str]:
    """The state (from the page title) and the value (the note's one number with a decimal point) of an answer."""
    state = title.rsplit(" Construction", 1)[0].rsplit(" construction", 1)[0]
    vals = re.findall(r"\d+\.\d", note)
    assert vals and len(set(vals)) == 1, (note, vals)
    return state, vals[0]


def clean(text: str) -> str:
    """The reply's first non-empty line that carries a number (else its first line), without quotes or a label."""
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines:
        return ""
    t = next((x for x in lines if re.search(r"\d", x)), lines[0])
    t = t.strip().strip('"').strip("“”").strip()
    return re.sub(r"^(rewritten note|note)\s*:\s*", "", t, flags=re.I).strip().strip('"')


def content_ok(new: str, state: str, value: str) -> bool:
    """Names the state and no other, and carries the value and no other number."""
    if not new or not re.search(rf"\b{re.escape(state)}\b", new):
        return False
    if any(s != state and s not in state and re.search(rf"\b{re.escape(s)}\b", new) for s in STATES):
        return False
    nums = NUM.findall(new)
    return bool(nums) and all(x == value for x in nums)


def length_ok(new: str, old: str) -> bool:
    return abs(len(new) - len(old)) <= max(15, 0.25 * len(old))


def rewrite(llm, note: str, title: str, seed: int) -> tuple[str | None, int, str]:
    """A rewrite of one note, the number of calls made, and how it passed: "ok", "approx" (right content, closest
    length after TRIES tries) or "dropped" (no rewrite with the right content: None)."""
    state, value = value_of(note, title)
    msg = [{"role": "user", "content": REWRITE.format(title=title, note=note, value=value, n=len(note))}]
    best = None
    for k in range(TRIES):
        new = clean(llm.chat(msg, temperature=0.7, max_tokens=150, seed=seed * 100 + k)["text"])
        if content_ok(new, state, value):
            if length_ok(new, note):
                return new, k + 1, "ok"
            if best is None or abs(len(new) - len(note)) < abs(len(best) - len(note)):
                best = new
    return best, TRIES, "approx" if best is not None else "dropped"


def build(llms: dict, src_dir: Path, out_dir: Path, workers: int = 32) -> dict:
    """Rewrite every source page with every writer in llms ({short: client}); returns per-file statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for source in MODELS:
        src = json.loads((src_dir / f"{source}_answers_q1.json").read_text())
        for writer, llm in llms.items():
            jobs = [(s, i, t, p["title"]) for s, p in src["pages"].items() for i, t in enumerate(p["notes"])]
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(lambda j: rewrite(llm, j[2], j[3], int(j[0]) * 1000 + j[1]), jobs))
            pages = {s: {"title": p["title"], "notes": []} for s, p in src["pages"].items()}
            for (s, i, t, _), (new, _, how) in zip(jobs, res):
                if new is not None:
                    pages[s]["notes"].append(new)
            how = [h for _, _, h in res]
            olen = [len(t) for _, _, t, _ in jobs]
            nlen = [len(n) for n, _, _ in res if n is not None]
            st = {"notes": len(jobs), "calls": sum(n for _, n, _ in res), "ok": how.count("ok"),
                  "approx": how.count("approx"), "dropped": how.count("dropped"),
                  "identical": sum(n == t for (_, _, t, _), (n, _, _) in zip(jobs, res)),
                  "mean_len_source": sum(olen) / len(olen), "mean_len_rewrite": sum(nlen) / max(1, len(nlen))}
            (out_dir / f"{source}_by_{writer}.json").write_text(json.dumps(
                {"source": f"results/page_content/{source}_answers_q1.json", "writer": WRITERS[writer],
                 "prompt": REWRITE, "tries": TRIES, "failed": [], "stats": st, "pages": pages},
                indent=1, ensure_ascii=False) + "\n")
            stats[f"{source}_by_{writer}"] = st
            print(f"{source}_by_{writer}: {len(jobs)} notes, {st['calls']} calls, {st['ok']} ok, {st['approx']} "
                  f"approx, {st['dropped']} dropped, {st['identical']} identical, mean length "
                  f"{st['mean_len_source']:.0f} -> {st['mean_len_rewrite']:.0f}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default=str(SC / "results" / "page_content"))
    ap.add_argument("--out", default=str(SC / "results" / "paraphrase"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--writers", nargs="+", default=["gemma", "q8"], choices=["gemma", "q8", "llama", "gptoss"])
    a = ap.parse_args()
    import os
    from scb.cli import _llm
    from scb.llm import LLM
    llms = {}
    for w in a.writers:
        if w == "llama":
            llms[w] = LLM(model=LLAMA, base_url=os.environ["SB_BASE_URL"], api_key=os.environ["SB_API_KEY"],
                          temperature=0.7, max_tokens=150,
                          extra_body={"provider": {"order": [LLAMA_HOST], "allow_fallbacks": False,
                                                   "require_parameters": True}})
        elif w == "gptoss":
            llms[w] = LLM(model=GPTOSS, base_url=os.environ["SB_BASE_URL"], api_key=os.environ["SB_API_KEY"],
                          temperature=0.7, max_tokens=150, token_mult=6.0,
                          extra_body={"provider": {"order": [GPTOSS_HOST], "allow_fallbacks": False,
                                                   "require_parameters": True}, "reasoning": {"effort": "low"}})
        else:
            llms[w] = _llm(MODELS[w], 0.7, 150)
    build(llms, Path(a.pages), Path(a.out), a.workers)


if __name__ == "__main__":
    random.seed(0)
    sys.exit(main())
