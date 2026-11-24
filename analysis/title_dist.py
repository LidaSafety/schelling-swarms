"""Round-1 title distributions of Qwen3-8B and Gemma 3 27B, and their title match M.

Usage: python title_dist.py <path to Swarm_schelling_hackathon/scaling/results>

Counts every round-1 write (holders and seekers alike) in the pure swarms of N = 2, 4, 8,
unaware text, rule in force. State names are replaced by <State>; upper and lower case count
as different titles. Reproduces mixtures.md: 1,330 and 1,396 titles, M = 0.303, 0.054, 0.640.
"""
import collections, gzip, json, re, sys

R = sys.argv[1].rstrip("/") + "/"


def template(path, states):
    for s in sorted(states, key=len, reverse=True):
        path = re.sub(re.escape(s), "<State>", path, flags=re.I)
    return path


def counts(files):
    c = collections.Counter()
    for f in files:
        for line in gzip.open(R + f, "rt"):
            g = json.loads(line)
            if g.get("awareness") != "unaware":
                continue
            for ev in g["events"].values():
                for w in ev["writes"]:
                    if w["round"] == 1:
                        c[template(w["path"], g["questions"])] += 1
    return c


Q = counts([f"swarm_main_n{n}/raw/qwen3-8b.jsonl.gz" for n in (2, 4, 8)])
G = counts([f"swarm_gemma_n{n}/raw/gemma-3-27b.jsonl.gz" for n in (2, 4, 8)])
nq, ng = sum(Q.values()), sum(G.values())
keys = sorted(set(Q) | set(G), key=lambda k: -(Q[k] / nq + G[k] / ng))
print(f"titles counted: Qwen {nq} ({len(Q)} distinct), Gemma {ng} ({len(G)} distinct)")
print(" p^Q_t   p^G_t   title t")
for k in keys:
    print(f"{Q[k] / nq:6.3f}  {G[k] / ng:6.3f}   {k}")
print(f"M_QQ = {sum((Q[k] / nq) ** 2 for k in keys):.3f}")
print(f"M_QG = {sum(Q[k] / nq * G[k] / ng for k in keys):.3f}")
print(f"M_GG = {sum((G[k] / ng) ** 2 for k in keys):.3f}")
