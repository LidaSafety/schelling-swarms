"""Check that every row of a committed law file is reproduced by a rerun (to 1e-9; the rerun sums in another order).

Usage: python check_law_rows.py <committed.json> <rerun.json>

The rerun may hold more rows than the committed file: scaling/results has gained runs since the committed files were
written (for example DeepSeek V4.1 Flash with 32 and 64 copies). The paper uses the committed files.
"""
import json
import sys


def close(u, v):
    if isinstance(u, float) and isinstance(v, float):
        return abs(u - v) < 1e-9
    if isinstance(u, list) and isinstance(v, list):
        return len(u) == len(v) and all(close(p, q) for p, q in zip(u, v))
    if isinstance(u, dict) and isinstance(v, dict):
        return u.keys() == v.keys() and all(close(u[k], v[k]) for k in u)
    return u == v


def key(row):
    return tuple(str(row.get(k)) for k in ("family", "target", "model", "rule"))


committed, rerun = (json.load(open(p)) for p in sys.argv[1:3])
by_key = {key(r): r for r in rerun}
bad = [key(r) for r in committed if key(r) not in by_key or not close(r, by_key[key(r)])]
name = sys.argv[1].rsplit("/", 1)[-1]
if bad:
    print(f"DIFFERS  {name}: {len(bad)} of {len(committed)} rows not reproduced, e.g. {bad[0]}")
else:
    print(f"same     {name}: all {len(committed)} rows reproduced ({len(rerun) - len(committed)} more rows in the rerun)")
