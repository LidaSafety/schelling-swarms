"""The title match of a swarm that mixes any number of models (4-round game).

The paper's Section 4.3 counts one title distribution per model, p^m (the round-1 titles its copies write to), and
the title match M[m][m'] = sum_t p^m_t p^m'_t, a Gram matrix. For a mix with shares x it has
    f(x) = sum_{m,m'} x_m x_m' M[m][m'] = ||p_bar||^2,      p_bar = sum_m x_m p^m,
and the gap below the weighted average of the pure swarms is
    sum_m x_m M[m][m] - f(x) = sum_m x_m ||p^m - p_bar||^2 = 1/2 sum_{m,m'} x_m x_m' d2[m][m'],
with d2[m][m'] = ||p^m - p^m'||^2 = M[m][m] + M[m'][m'] - 2 M[m][m']. This script counts M for the five models,
checks it against the paper's Table 2 and the four mixed swarms of its Table 6, works the three-model case for the
figure of Appendix F, and checks the identities numerically. Writes mixtures.md and out/mixtures.json.

Reads the logs with fuller.py's reader. Counted after the runs, not registered. Rerun with `python mixtures.py`.
"""
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / "fuller.py").read_text()
g = {"__file__": str(HERE / "fuller.py"), "__name__": "fuller_part"}
exec(compile(src[:src.index("# --------------------------------------------------------------------------------------------- the held-out cells")],
             "fuller.py", "exec"), g)
rows, SMALL, SRC = g["rows"], g["SMALL"], g["SRC"]
SHORT = {"qwen3-4b": "4B", "qwen3-8b": "8B", "qwen3-14b": "14B", "qwen3-32b-bf16": "32B", "gemma-3-27b": "Gemma"}
MODELS = list(SHORT)
lines, J = [], {}


def say(s=""):
    print(s)
    lines.append(s)


def dot(p, q):
    return sum(v * q.get(t, 0.0) for t, v in p.items())


# ------------------------------------------------------------------------------------------------- 1. the counts
P, n = {}, {}
for m in MODELS:
    c = Counter(x["w_tpl"] for x in rows if x["set"] == SRC[m] and x["model"] == m and x["round"] == 1
                and x["N"] <= SMALL and x["w_tpl"])
    n[m] = sum(c.values())
    P[m] = {t: v / n[m] for t, v in c.items()}
M = np.array([[dot(P[a], P[b]) for b in MODELS] for a in MODELS])
D2 = np.array([[M[i, i] + M[j, j] - 2 * M[i, j] for j in range(len(MODELS))] for i in range(len(MODELS))])

say("# The title match of a swarm that mixes any number of models\n")
say("One title distribution per model, p^m: the round-1 titles its copies write to, holders and seekers alike, on its "
    f"pure swarms of N ≤ {SMALL} (Qwen3-8B from the main runs, Gemma 3 27B from its own, the other Qwen3 sizes from "
    "the size comparison). State names are replaced by a placeholder, as in the paper. Rerun with `python mixtures.py`.\n")
say("## 1. The title match M of the five models\n")
say("| | " + " | ".join(SHORT[m] for m in MODELS) + " |")
say("|---|" + "---|" * len(MODELS))
for i, a in enumerate(MODELS):
    say(f"| {SHORT[a]} | " + " | ".join(f"{M[i, j]:.3f}" for j in range(len(MODELS))) + " |")
say("| Titles counted | " + " | ".join(f"{n[m]:,}" for m in MODELS) + " |")
say()
say(f"Qwen3-8B and Gemma 3 27B: {M[1, 1]:.3f}, {M[1, 4]:.3f}, {M[4, 4]:.3f}; the paper's Table 2 has 0.30, 0.054 and "
    "0.64.\n")
say("Squared distance d² = M[a][a] + M[b][b] − 2 M[a][b] for each pair; an even mix of the two sags by d²/4:\n")
say("| | " + " | ".join(SHORT[m] for m in MODELS) + " |")
say("|---|" + "---|" * len(MODELS))
for i, a in enumerate(MODELS):
    say(f"| {SHORT[a]} | " + " | ".join("–" if i == j else f"{D2[i, j]:.3f}" for j in range(len(MODELS))) + " |")
say()
J["models"] = [SHORT[m] for m in MODELS]
J["M"] = M.tolist()
J["d2"] = D2.tolist()
J["titles"] = [n[m] for m in MODELS]

# ------------------------------------------------------------------------------------ 2. the mixed swarms that ran
say("## 2. The four mixed swarms of the paper (Qwen3-8B with Gemma 3 27B)\n")
say("*Pooled*: any two round-1 titles of the mix, without a title paired with itself. *Same game*: round-1 titles of "
    "two different copies of one game. f(x) predicts the first, f_N(x) = f(x) − gap/(N − 1) the second.\n")
say("| Swarm | Gemma share x | f(x) | Measured, pooled | f_N(x) | Measured, same game | Weighted average |")
say("|---|---|---|---|---|---|---|")
J["mixes"] = []
Q, G = 1, 4
for nq, ng in ((2, 2), (6, 2), (4, 4), (2, 6)):
    N = nq + ng
    lab = f"qwen3-8b:{nq},gemma-3-27b:{ng}"
    R = [x for x in rows if x["set"] == "mix" and x["mix"] == lab and x["round"] == 1 and x["w_tpl"]]
    c = Counter(x["w_tpl"] for x in R)
    k = sum(c.values())
    pooled = (sum(v * v for v in c.values()) - k) / (k * (k - 1))
    games = defaultdict(list)
    for x in R:
        games[x["game"]].append(x["w_tpl"])
    pairs = [ts[i] == ts[j] for ts in games.values() for i, j in itertools.combinations(range(len(ts)), 2)]
    same = float(np.mean(pairs))
    x = ng / N
    lin = (1 - x) * M[Q, Q] + x * M[G, G]
    f = (1 - x) ** 2 * M[Q, Q] + 2 * x * (1 - x) * M[Q, G] + x ** 2 * M[G, G]
    fN = f - (lin - f) / (N - 1)
    J["mixes"].append(dict(swarm=f"{nq}Q+{ng}G", x=x, f=f, pooled=pooled, fN=fN, same=same, avg=lin))
    say(f"| {nq}Q+{ng}G | {x:g} | {f:.3f} | {pooled:.3f} | {fN:.3f} | {same:.3f} | {lin:.3f} |")
err = [abs(r["f"] - r["pooled"]) for r in J["mixes"]] + [abs(r["fN"] - r["same"]) for r in J["mixes"]]
say(f"\nLargest difference between prediction and measurement: {max(err):.3f}.\n")

# ----------------------------------------------------------------------------------------------- 3. three models
say("## 3. Three models: Qwen3-4B, Qwen3-8B and Gemma 3 27B (the figure of Appendix F)\n")
I3 = [0, 1, 4]
M3 = M[np.ix_(I3, I3)]
d3 = D2[np.ix_(I3, I3)]
grid = [(a / 60, b / 60, 1 - a / 60 - b / 60) for a in range(61) for b in range(61 - a)]
vals = [(np.array(x) @ M3 @ np.array(x), x) for x in grid]
fmin, xmin = min(vals)
centre = np.ones(3) / 3
gap_c = np.diag(M3) @ centre - centre @ M3 @ centre
pair_best = min(min(((1 - s) ** 2 * M3[i, i] + 2 * s * (1 - s) * M3[i, j] + s ** 2 * M3[j, j], s)
                    for s in np.linspace(0, 1, 601)) for i, j in itertools.combinations(range(3), 2))
say(f"Corners (pure swarms): {M3[0, 0]:.3f}, {M3[1, 1]:.3f}, {M3[2, 2]:.3f}. Squared distances: 4B–8B "
    f"{d3[0, 1]:.3f}, 4B–Gemma {d3[0, 2]:.3f}, 8B–Gemma {d3[1, 2]:.3f}.\n")
say(f"Even three-way mix: f = {centre @ M3 @ centre:.3f} against a weighted average of {np.diag(M3).mean():.3f}; the "
    f"gap is {gap_c:.3f} = (sum of the three d²)/9 = {d3[np.triu_indices(3, 1)].sum() / 9:.3f}.\n")
say(f"Lowest title match over all mixes of the three: {fmin:.3f} at shares 4B {xmin[0]:.2f}, 8B {xmin[1]:.2f}, "
    f"Gemma {xmin[2]:.2f}. The lowest over mixes of two of them is {pair_best[0]:.3f}.\n")
J["three"] = dict(models=["4B", "8B", "Gemma"], M=M3.tolist(), d2=d3.tolist(), centre_f=float(centre @ M3 @ centre),
                  centre_avg=float(np.diag(M3).mean()), centre_gap=float(gap_c), fmin=float(fmin),
                  xmin=list(map(float, xmin)), pair_min=float(pair_best[0]))

# ------------------------------------------------------------------------------------------------- 4. identities
rng = np.random.default_rng(3)
worst = 0.0
for _ in range(2000):
    k, T = int(rng.integers(2, 7)), int(rng.integers(2, 12))
    p, x = rng.dirichlet(np.ones(T) * 0.5, k), rng.dirichlet(np.ones(k))
    Mg = p @ p.T
    gap = x @ np.diag(Mg) - x @ Mg @ x
    pbar = x @ p
    var = sum(x[i] * ((p[i] - pbar) ** 2).sum() for i in range(k))
    half = 0.5 * sum(x[i] * x[j] * ((p[i] - p[j]) ** 2).sum() for i in range(k) for j in range(k))
    worst = max(worst, abs(gap - var), abs(gap - half), abs(x @ Mg @ x - pbar @ pbar))
say("## 4. Check of the identities\n")
say(f"On 2,000 random sets of 2 to 6 distributions over 2 to 11 titles, the largest violation of "
    f"f = ||p̄||² and gap = variance = half the weighted sum of squared distances is {worst:.1e}.")
J["check"] = worst

(HERE / "mixtures.md").write_text("\n".join(lines) + "\n")
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "mixtures.json").write_text(json.dumps(J, indent=1))
