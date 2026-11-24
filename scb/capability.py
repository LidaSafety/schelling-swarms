"""A capability score for each model, measured with the settings it plays the game with: a fixed multiple-choice quiz
(the MMLU-Pro subset drawn by scripts/mmlupro_subset.py, 504 questions, 10 options each), one question per call,
through the same client and thinking setting as scb swarm. As in MMLU-Pro's own evaluation, a model may work the
question out in its reply and ends with "Answer: X"; max_tokens is 2048 (times the model's token_mult). A first try
that asked for the letter alone within the game's 250 tokens measured truncation and format, not capability: many
models reason in text regardless. Published
scores are not on one scale for these models (different index versions and reasoning settings), so this gives the
rule ablation one axis. Each call's seed is the run seed * 100000 + the question's index (below 2^31, which some
OpenRouter hosts require)."""
from __future__ import annotations

import json
import re
from typing import Any

SYSTEM = ("Answer the multiple-choice question. Think briefly if you need to, then end your reply with a final line "
          '"Answer: X", where X is the letter of the correct option.')
MAX_TOKENS = 2048
LETTERS = "ABCDEFGHIJ"
_STATED = re.compile(r"[Aa][Nn][Ss][Ww][Ee][Rr]\W{0,4}(?:is\W{0,4})?\(?([A-J])\b")    # "Answer: C", "answer is (C)", "**Answer:** C"
_BARE = re.compile(r"^\W*\(?([A-J])(?=\W*$|[).:\]*,])")                   # a line that is just "C", "(C)", "**C**"


def prompt(q: dict[str, Any]) -> list[dict[str, str]]:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{q['question']}\n\n{opts}"}]


def parse(text: str) -> str | None:
    """The chosen letter: the last "Answer: X", else the last line that is a bare letter, else a first line that is."""
    t = (text or "").strip()
    found = _STATED.findall(t)
    if found:
        return found[-1]
    lines = [x for x in t.splitlines() if x.strip()]
    for line in reversed(lines[-1:]) if lines else []:
        m = _BARE.match(line.strip())
        if m:
            return m.group(1)
    m = _BARE.match(lines[0].strip()) if lines else None
    return m.group(1) if m else None


def ask(llm, q: dict[str, Any], i: int, seed: int, max_tokens: int) -> dict[str, Any]:
    try:
        out = llm.chat(prompt(q), max_tokens=max_tokens, seed=seed * 100000 + i)
    except Exception as e:  # noqa: BLE001
        return {"qid": q["qid"], "category": q["category"], "answer": q["answer"], "error": str(e)[:300]}
    pred = parse(out["text"])
    return {"qid": q["qid"], "category": q["category"], "answer": q["answer"], "reply": out["text"][-3000:],
            "pred": pred, "correct": pred == q["answer"]}


def load(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list[dict[str, Any]], b: int = 2000, seed: int = 0) -> dict[str, Any]:
    import random
    ok = [r for r in rows if "error" not in r]
    acc = [1.0 if r["correct"] else 0.0 for r in ok]
    if not acc:
        return {"n": 0, "errors": len(rows)}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(acc) for _ in acc) / len(acc) for _ in range(b))
    return {"n": len(ok), "errors": len(rows) - len(ok), "accuracy": sum(acc) / len(acc),
            "ci95": [boots[int(0.025 * b)], boots[int(0.975 * b) - 1]],
            "unparsed": sum(r["pred"] is None for r in ok)}
