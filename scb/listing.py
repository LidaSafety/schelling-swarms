"""Top-level directory names shown for the dead drop's shared drive (listing=workspace or random)."""
from __future__ import annotations

import random

# Plausible directories on a shared research / dev drive.
WORKSPACE_POOL = [
    "archive", "assets", "backups", "build", "cache", "configs", "data", "datasets",
    "docs", "downloads", "experiments", "exports", "figures", "imports", "logs",
    "misc", "models", "notebooks", "notes", "old", "outputs", "public", "raw",
    "reports", "results", "scratch", "scripts", "shared", "src", "staging", "tests",
    "tmp", "tools", "uploads", "vendor", "wiki", "work",
]


def _pseudo_word(rng: random.Random, n: int) -> str:
    cons = "bcdfghjklmnprstvwz"
    vow = "aeiou"
    s = ""
    for i in range(n):
        s += rng.choice(cons) if i % 2 == 0 else rng.choice(vow)
    return s


def top_dirs(preset: str, seed: int, k: int = 12) -> list[str]:
    """`k` directory names, sorted: a sample of WORKSPACE_POOL (preset=workspace) or pseudo-words (preset=random)."""
    rng = random.Random(seed)
    if preset == "workspace":
        dirs = rng.sample(list(WORKSPACE_POOL), min(k, len(WORKSPACE_POOL)))
    elif preset == "random":
        dirs = []
        while len(dirs) < k:
            w = _pseudo_word(rng, rng.choice([4, 5, 6, 7]))
            if w not in dirs:
                dirs.append(w)
    else:
        raise ValueError(f"unknown preset {preset!r}")
    return sorted(dirs)
