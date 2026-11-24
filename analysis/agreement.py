#!/usr/bin/env python3
"""Seekers' use of a found true value by what else the pages they read show: DeepSeek V4.1 Flash (no rule) and
Qwen3-4B (rule in force), 64 copies each.

  python agreement.py <scaling dir of the scaling branch> [--reps 200] > agreement.md

Reuses notes_of, judge and DEC of analyze_posting.py on the scaling branch unchanged. For every seeker chance (a copy
answering a question it does not hold), the pages it read in rounds up to the question's round are taken as the
read events log them (path and the authors on it at the time). Each other author on such a page counts once:
  R     authors with a note that carries the true value (judge == "right"),
  W     the other authors with a note that carries a wrong value for that question (judge == "wrong"),
  Wmax  the largest group of those W authors whose wrong notes carry the same decimals (DEC).
Used: the copy's answer is a value it found on the wiki (used_chosen, as in analyze_ablations.py). Proportional: the
mean of R / (R + W), the use that the prediction from pairs assumes beside wrong values. 95% intervals resample games.
"""
import argparse, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scaling")
ap.add_argument("--reps", type=int, default=200)
a = ap.parse_args()
sc = Path(a.scaling).resolve()
sys.path.insert(0, str(sc / "scripts")); sys.path.insert(0, str(sc.parent))
from analyze_posting import DEC, judge, notes_of  # noqa: E402

RUNS = (("DeepSeek V4.1 Flash, no rule", "swarm_deepseek_large/raw/deepseek_deepseek-v4.1-flash.jsonl.gz"),
        ("Qwen3-4B, rule in force", "swarm_scale_n64/raw/qwen3-4b.jsonl.gz"))
N = 64


def chances(r):
    """One dict per seeker chance of game r."""
    truth = {x["state"]: x["truth"] for ag in r["agents"].values() for x in ag["answers"]}
    notes = defaultdict(list)                                  # (author, title) -> [(round, text, title)]
    for rk, au, t, x in notes_of(r):
        notes[(au, t)].append((rk, x, t))
    out = []
    for c, ag in r["agents"].items():
        for x in ag["answers"]:
            if not x["chance"]:
                continue
            st, k = x["state"], x["round"]
            R, Wv = set(), {}
            for rd in r["events"][c]["reads"]:
                if rd["round"] > k:
                    continue
                for au in rd["authors"]:
                    if au == c:
                        continue
                    for rk, txt, t in notes[(au, rd["path"])]:
                        if rk is None or rk > rd["round"]:
                            continue
                        j = judge(t, txt, st, truth)
                        if j == "right":
                            R.add(au)
                        elif j == "wrong":
                            Wv.setdefault(au, frozenset(DEC.findall(txt)))
            W = {au: v for au, v in Wv.items() if au not in R}
            groups = Counter(W.values())
            out.append({"k": k, "R": len(R), "W": len(W), "Wmax": max(groups.values(), default=0),
                        "Wk": len(groups), "seen": bool(x["seen"]),
                        "used": int(x.get("used_chosen", x["used_other"]))})
    return out


def load(path):
    games = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["n_agents"] == N:
                games.append(chances(r))
    return games


def row(games, keep, boots):
    """Chances, used %, its 95% interval, proportional % over the chances that pass keep."""
    per = np.array([[sum(1 for x in g if keep(x)), sum(x["used"] for x in g if keep(x)),
                     sum(x["R"] / (x["R"] + x["W"]) for x in g if keep(x))] for g in games])
    t = per.sum(0)
    if t[0] == 0:
        return None
    ci = np.percentile([per[b].sum(0)[1] / max(per[b].sum(0)[0], 1) for b in boots], [2.5, 97.5])
    return int(t[0]), 100 * t[1] / t[0], 100 * ci, 100 * t[2] / t[0]


def fmt(x):
    if x is None:
        return "| 0 | - | - |"
    n, u, ci, p = x
    return f"| {n:,} | {u:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {p:.1f} |"


print("# Seekers' use of a found true value, by agreement on the page\n")
print("Written by `agreement.py` (23 November) from the 64-copy games of the scaling branch of the hackathon repository")
print("(commit a3709a3): `results/swarm_deepseek_large` (DeepSeek V4.1 Flash without the rule, the runs of app:deepseek)")
print("and `results/swarm_scale_n64` (Qwen3-4B with the rule, the runs of fig:law). It reuses `notes_of`, `judge` and")
print("`DEC` of `scripts/analyze_posting.py` unchanged. For each seeker chance: R = other copies whose notes on the pages")
print("read by the question's round carry the true value; W = other copies whose notes carry a wrong value for it;")
print("Wmax = the largest group of those copies that wrote the same wrong number. Used = the answer is a value found on")
print("the wiki. Proportional = mean R/(R+W), what the prediction from pairs assumes beside wrong values (sec:law).")
print("Rates pool chances; 95% intervals resample games (" + str(a.reps) + " draws).\n")

rng = np.random.default_rng(0)
for name, rel in RUNS:
    games = load(sc / "results" / rel)
    boots = [rng.integers(0, len(games), len(games)) for _ in range(a.reps)]
    allc = [x for g in games for x in g]
    found = [x for x in allc if x["R"] >= 1]
    both = [x for x in found if x["W"] >= 1]
    print(f"## {name}, {N} copies ({len(games)} games)\n")
    print(f"Chances: {len(allc):,}; with a true value read: {len(found):,}. R >= 1 agrees with the logged `seen` flag "
          f"in {100 * sum((x['R'] >= 1) == x['seen'] for x in allc) / len(allc):.1f}% of chances. Beside at least one "
          f"wrong value: {len(both):,} chances, with on average {np.mean([x['R'] for x in both]):.1f} true notes, "
          f"{np.mean([x['W'] for x in both]):.1f} wrong notes, {np.mean([x['Wk'] for x in both]):.1f} distinct wrong "
          f"values and a largest wrong group of {np.mean([x['Wmax'] for x in both]):.1f}; the true value is the most "
          f"common value in {100 * np.mean([x['R'] > x['Wmax'] for x in both]):.1f}% of them.\n")

    print("**Is the true value the most common value on the page?** (chances with R >= 1 and W >= 1)\n")
    print("| Page | Chances | Used, % [95%] | Proportional, % |\n|---|---|---|---|")
    for lab, f in (("True value most common (R > Wmax)", lambda x: x["R"] > x["Wmax"]),
                   ("... although wrong notes outnumber true ones (W > R)", lambda x: x["R"] > x["Wmax"] and x["W"] > x["R"]),
                   ("Tied with a wrong value (R = Wmax)", lambda x: x["R"] == x["Wmax"]),
                   ("A wrong value more common (R < Wmax)", lambda x: x["R"] < x["Wmax"])):
        print(f"| {lab} " + fmt(row(games, lambda x, f=f: x["R"] >= 1 and x["W"] >= 1 and f(x), boots)))
    print()

    print("**By the true value's lead over the largest wrong group, R - Wmax** (chances with R >= 1 and W >= 1)\n")
    print("| R - Wmax | Chances | Used, % [95%] | Proportional, % |\n|---|---|---|---|")
    for lo, hi, lab in ((-999, -2, "-2 or less"), (-1, -1, "-1"), (0, 0, "0"), (1, 1, "+1"), (2, 3, "+2 to +3"),
                        (4, 999, "+4 or more")):
        print(f"| {lab} " + fmt(row(games, lambda x, lo=lo, hi=hi: x["R"] >= 1 and x["W"] >= 1
                                     and lo <= x["R"] - x["Wmax"] <= hi, boots)))
    print()

    print("**With no wrong value on the page, by true notes R**\n")
    print("| R | Chances | Used, % [95%] | Proportional, % |\n|---|---|---|---|")
    for lo, hi, lab in ((1, 1, "1"), (2, 2, "2"), (3, 999, "3 or more")):
        print(f"| {lab} " + fmt(row(games, lambda x, lo=lo, hi=hi: x["W"] == 0 and lo <= x["R"] <= hi, boots)))
    print()

    print("**By round half** (all chances with R >= 1)\n")
    print("| Rounds | Chances | Used, % [95%] | Proportional, % |\n|---|---|---|---|")
    for lab, ks in (("1 and 3", (1, 3)), ("2 and 4", (2, 4))):
        print(f"| {lab} " + fmt(row(games, lambda x, ks=ks: x["R"] >= 1 and x["k"] in ks, boots)))
    print()
