"""Compare a rerun JSON file with the committed one, allowing float differences below 1e-9 (summation order).

Usage: python check_json.py <committed.json> <rerun.json>
Prints `same` or `DIFFERS` with the path of the first difference.
"""
import json
import math
import sys


def first_diff(u, v, path="$"):
    if isinstance(u, float) and isinstance(v, (int, float)) or isinstance(v, float) and isinstance(u, (int, float)):
        if math.isnan(u) and math.isnan(v):
            return None
        return None if abs(u - v) < 1e-9 else f"{path}: {u} vs {v}"
    if type(u) is not type(v):
        return f"{path}: {type(u).__name__} vs {type(v).__name__}"
    if isinstance(u, dict):
        for k in u.keys() | v.keys():
            if k not in u or k not in v:
                return f"{path}.{k}: only in {'committed' if k in u else 'rerun'}"
            d = first_diff(u[k], v[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(u, list):
        if len(u) != len(v):
            return f"{path}: {len(u)} vs {len(v)} items"
        for i, (p, q) in enumerate(zip(u, v)):
            d = first_diff(p, q, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if u == v else f"{path}: {u!r} vs {v!r}"


committed, rerun = (json.load(open(p)) for p in sys.argv[1:3])
d = first_diff(committed, rerun)
name = sys.argv[1].rsplit("/", 1)[-1]
print(f"same     {name}" if d is None else f"DIFFERS  {name}: {d}")
