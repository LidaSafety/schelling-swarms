"""The paper's figures extended to the larger swarms of pod H (N = 64 and 128), with the cells they join, all read from
the raw game records in one way.

For each pure swarm (the Qwen3 sizes, and Gemma 3 27B at N = 2-8 for the per-model figure): CR over all rounds and over round 1 alone (95% intervals, resampling whole games), and the
steps of Table 7 and Figure 4's table: holders who post their value, seekers who find it, seekers who use what they
found, re-posts of others' values per question, and read title matches (round 1: the chance that the title a copy
reads is the title another copy, drawn at random, wrote to). Cells already in the paper are recomputed as a check on the definitions. Writes larger_n.md
and out/larger_n.json. Rerun with `python larger_n.py`.
"""
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import ROOT, SCALING, STAGE1, RES  # noqa: E402
REF = RES / "swarm4round"
B = 2000
rng = np.random.default_rng(20261118)

# (model, f_inv) -> {N: run folder}
CELLS = {
    ("qwen3-8b", 2): {**{n: STAGE1 / f"swarm_main_n{n}" for n in (2, 4, 8, 16, 32)},
                      64: REF / "swarm_ref_8b_n64", 128: RES / "swarm_scale_n128"},
    ("qwen3-8b", 4): {**{n: STAGE1 / f"swarm_comp_n{n}" for n in (4, 8, 16, 32)},
                      64: RES / "swarm_scale1_n64", 128: RES / "swarm_scale1_n128"},
    **{(m, 2): {**{n: REF / f"swarm_ref_{s}" for n in (2, 4, 8, 16, 32)},
                64: RES / "swarm_scale_n64", 128: RES / "swarm_scale_n128"}
       for m, s in (("qwen3-4b", "4b"), ("qwen3-14b", "14b"), ("qwen3-32b-bf16", "32b"))},
    # Gemma 3 27B alone: stage 1 at N = 2-8, pod I (benchmark entry 21) at N = 16-128
    ("gemma-3-27b", 2): {**{n: STAGE1 / f"swarm_gemma_n{n}" for n in (2, 4, 8)},
                         **{n: RES / f"swarm_gemma_ladder_n{n}" for n in (16, 32, 64, 128)}},
}
# Pod I's control: Gemma at N = 8 again on the new GPU, beside stage 1's N = 8 above
CONTROLS = {("gemma-3-27b", 2, 8): RES / "swarm_gemma_ladder_n8"}


def load(d, model, n, f_inv):
    out = []
    for f in sorted((d / "raw").glob(f"{model}.jsonl*")):
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if ("error" in r or r.get("n_agents") != n or r.get("f_inv") != f_inv or r.get("rule") != "forbidden"
                        or r.get("awareness") != "unaware" or r.get("plant") not in (None, "none")):
                    continue
                out.append(r)
    return out


def per_game(r):
    """Counts for one game: chances, uses, round-1 chances and uses, seen, uses of seen, holder-rounds, posts,
    re-posts, questions, round-1 reads, and for each round-1 read the share of the other copies whose round-1 title it
    matches (the paper's "read title matches"; within one round all copies face the same state, so exact titles
    compare as titles with the state replaced)."""
    c = dict.fromkeys(("ch", "use", "ch1", "use1", "seen", "useseen", "hold", "post", "repost", "q", "rd1", "match1"), 0)
    hold = {a: set(v) for a, v in r["holds"].items()}
    w1 = {a: next((w["path"] for w in ev["writes"] if w["round"] == 1 and w.get("path")), None)
          for a, ev in r["events"].items()}
    for a, ag in r["agents"].items():
        ev = r["events"][a]
        posts = {w["round"]: set(w.get("posts") or []) for w in ev["writes"]}
        for x in ag["answers"]:
            rd, st = x["round"], x["state"]
            if st in hold[a]:
                c["hold"] += 1
                c["post"] += st in posts.get(rd, ())
            else:
                c["repost"] += st in posts.get(rd, ())
            if x["chance"]:
                u = bool(x["used_chosen"])
                c["ch"] += 1
                c["use"] += u
                c["seen"] += bool(x["seen"])
                c["useseen"] += u and bool(x["seen"])
                if rd == 1:
                    c["ch1"] += 1
                    c["use1"] += u
        for x in ev["reads"]:
            if x["round"] == 1 and x.get("path"):
                c["rd1"] += 1
                others = [p for b, p in w1.items() if b != a and p]
                c["match1"] += sum(p == x["path"] for p in others) / len(others) if others else 0
    c["q"] = len(r["questions"]) if isinstance(r.get("questions"), list) else r.get("n_questions", 4)
    return c


def rate(G, num, den):
    pt = 100 * G[num].sum() / G[den].sum()
    idx = rng.integers(0, len(G[num]), (B, len(G[num])))
    d = 100 * G[num][idx].sum(1) / G[den][idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(pt=pt, lo=lo, hi=hi)


out, lines = {}, []
say = lines.append
say("# The paper's figures at N = 64 and 128\n")
say("Pure swarms (Qwen3, and Gemma 3 27B up to N = 8), rule in force, unaware, seed 1 (games 10000–). Pod H ran N = 64 and 128 with 100 and 50 games; "
    "the earlier cells are recomputed here from their raw records as a check. Percent unless noted; brackets are "
    f"95% intervals from {B:,} resamples of whole games. Rerun with `python larger_n.py`.\n")
say("| Model | Answers per copy | N | h | Games | CR | CR, round 1 | Holders post | Seekers find | Use what found | "
    "Re-posts per question | Read title matches |")
say("|---|---|---|---|---|---|---|---|---|---|---|---|")
def row(m, fi, n, d, key):
    games = load(d, m, n, fi)
    if not games:
        return
    rows = [per_game(r) for r in games]
    G = {k: np.array([x[k] for x in rows], float) for k in rows[0]}
    cell = dict(model=m, f_inv=fi, N=n, h=n // fi, games=len(games), cr=rate(G, "use", "ch"),
                cr1=rate(G, "use1", "ch1"), post=100 * G["post"].sum() / G["hold"].sum(),
                find=100 * G["seen"].sum() / G["ch"].sum(), usefound=100 * G["useseen"].sum() / G["seen"].sum()
                if G["seen"].sum() else float("nan"), repost=G["repost"].sum() / G["q"].sum(),
                match=G["match1"].sum() / G["rd1"].sum() if G["rd1"].sum() else float("nan"))
    out[key] = cell
    f = lambda r: f"{r['pt']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]"  # noqa: E731
    say(f"| {m} | {4 // fi} | {n} | {n // fi} | {len(games)} | {f(cell['cr'])} | {f(cell['cr1'])} | {cell['post']:.1f} | "
        f"{cell['find']:.1f} | {cell['usefound']:.1f} | {cell['repost']:.2f} | {cell['match']:.2f} |")


for (m, fi), runs in CELLS.items():
    for n, d in sorted(runs.items()):
        row(m, fi, n, d, f"{m}|{fi}|{n}")
head = lines[2:4]
say("\n## Controls\n")
say("The same cell run again on another GPU, to set beside the row above that it repeats.\n")
lines.extend(head)
for (m, fi, n), d in CONTROLS.items():
    row(m, fi, n, d, f"control|{m}|{fi}|{n}")
(HERE / "larger_n.md").write_text("\n".join(lines) + "\n")
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "larger_n.json").write_text(json.dumps(out, indent=1))
print("\n".join(lines))
