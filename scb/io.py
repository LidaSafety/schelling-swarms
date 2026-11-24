"""JSONL files, gzipped when the name ends in .gz (raw runs with full transcripts compress ~25x)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(p: str | Path) -> list[dict[str, Any]]:
    p = Path(p)
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.name.endswith(".gz") else open
    with opener(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
