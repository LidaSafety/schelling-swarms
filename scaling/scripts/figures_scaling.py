#!/usr/bin/env python3
"""Figures for the scaling study, from scripts/predict_swarm.py's output and the raw runs.

  .venv/bin/python scripts/predict_swarm.py results     # writes results/predict_swarm.json
  .venv/bin/python scripts/figures_scaling.py results [--width 5.5]

Writes results/figures/fig*.pdf (vector, fonts embedded) and .png previews at the report's text width (5.5 in, a
single-column layout; --width to match the template). One encoding throughout: unaware text = blue circles, aware
text = green triangles, twin text ("copies of you"; only in rows from before commit 107ae64, which removed
it) = vermillion squares; observed = solid line,
markers, 95% intervals (hollow at N=2 where the told wording differs from N>=4); predicted = dashed line and a
light band, from pairs (not told) or from N=4 runs (told arms, --aware-base 4 in predict_swarm.py).
  fig1_cr      collusion per chance against N, observed and predicted, one panel per condition
  fig2_found   the share of chances where the missing value had been seen, observed and predicted
  fig3_twin    twin text - aware text at each N: observed (paired by seed, scb.swarm.cluster_diff) and predicted
  fig4_pany    runs in which some copy used another's value
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_swarm import load_rows  # noqa: E402
from scb.swarm import _boot, cluster_diff, cr, pair_runs  # noqa: E402

# family suffixes, as scripts/predict_swarm.py family() names them
UNAWARE, AWARE = "unaware/none/N*-Q4-finv2-R0-log-none", "aware/none/N*-Q4-finv2-R0-log-none"
TWIN = AWARE + "_twin"
COND = {  # family suffix -> (color, marker, label)
    UNAWARE: ("#0072B2", "o", "unaware text"),
    AWARE: ("#009E73", "^", "aware text"),
    TWIN: ("#D55E00", "s", "twin text"),
}
GRAY, INK = "#6f6f6f", "#222222"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})


def pts(fam: dict, which: str, metric: str) -> list[tuple]:
    out = []
    for n, v in fam.items():
        if v.get(which):
            out.append((int(n), *(100 * x for x in v[which][metric])))
    return sorted(out)


def n_axis(ax, ns: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.set_xlim(ns[0] / 1.25, ns[-1] * 1.25)
    ax.set_xlabel("Copies in the swarm (N)")


def observed(ax, p: list[tuple], color: str, marker: str) -> None:
    if p:
        ax.errorbar([x[0] for x in p], [x[1] for x in p], yerr=[[x[1] - x[2] for x in p], [x[3] - x[1] for x in p]],
                    color=color, marker=marker, ms=3.8, lw=1.3, elinewidth=0.8, capsize=0, zorder=3)


def predicted(ax, p: list[tuple], color: str, band: bool = True) -> None:
    if p:
        x = [q[0] for q in p]
        if band:
            ax.fill_between(x, [q[2] for q in p], [q[3] for q in p], color=color, alpha=0.14, lw=0, zorder=1)
        ax.plot(x, [q[1] for q in p], ls=(0, (4, 2)), color=color, lw=1.0, zorder=2)


def end_labels(ax, items: list[tuple[list[tuple], str, str]]) -> None:
    lo, hi = ax.get_ylim()
    gap = 0.075 * (hi - lo)
    rows = sorted(([p[-1][0], p[-1][1], p[-1][1], t, c] for p, t, c in items if p), key=lambda r: r[1])
    for i in range(1, len(rows)):
        rows[i][2] = max(rows[i][2], rows[i - 1][2] + gap)
    shift = max(0.0, rows[-1][2] - hi) if rows else 0.0
    for x, y, ya, t, c in rows:
        ax.annotate(t, (x, y), xytext=(x * 1.12, ya - shift), textcoords="data", va="center", fontsize=7, color=c,
                    annotation_clip=False)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--prefix", default="wiki/forbidden/", help="channel/rule/ of the families to plot")
    ap.add_argument("--width", type=float, default=5.5)
    a = ap.parse_args()
    setup()
    res = Path(a.results)
    pred = json.loads((res / "predict_swarm.json").read_text())
    fams = {suf: pred["families"].get(f"{a.model} {a.prefix}{suf}") for suf in COND}
    fams = {k: v for k, v in fams.items() if v}
    if not fams:
        raise SystemExit(f"no families for {a.model} {a.prefix}* in {res / 'predict_swarm.json'}")
    out = res / "figures"
    out.mkdir(exist_ok=True)
    ns = sorted({int(n) for f in fams.values() for n in f})

    rows = [r for r in load_rows(res)
            if r.get("model") == a.model and f"{r.get('channel')}/{r.get('rule')}/" == a.prefix and "error" not in r
            and r.get("n_questions") == 4 and r.get("f_inv") == 2 and r.get("defense", "none") == "none"
            and not r.get("randomized") and r["log"] == "none" and r.get("listing") == "none" and not r.get("read_cap")]

    # 1: collusion per chance, observed against predicted, one panel per condition
    fig, axes = plt.subplots(1, len(fams), figsize=(a.width, a.width * 0.36), sharey=True, squeeze=False)
    for k, (ax, (suf, fam)) in enumerate(zip(axes[0], fams.items())):
        c, m, lab = COND[suf]
        pr, ob = pts(fam, "predicted", "cr"), pts(fam, "observed", "cr")
        predicted(ax, pr, c)
        observed(ax, ob, c, m)
        if pr and min(p[0] for p in pr) > 2:   # base above N=2: the N=2 wording differs, so show N=2 hollow, unjoined
            tw = 1 if suf == TWIN else 0
            g = [r for r in rows if r["n_agents"] == 2 and r["awareness"] == "aware" and r["twin"] == tw]
            if g:
                lo, hi = _boot(g, cr)
                ax.errorbar([2], [100 * cr(g)], yerr=[[100 * (cr(g) - lo)], [100 * (hi - cr(g))]], color=c, marker=m,
                            mfc="white", ms=3.8, lw=0, elinewidth=0.8, capsize=0, zorder=3)
        n_axis(ax, ns)
        ax.set_ylim(0, 100)
        ax.text(0.0, 1.03, f"{'abc'[k]}  {lab}", transform=ax.transAxes, fontsize=8, va="bottom", color=INK)
        if k == 0:
            ax.set_ylabel("Chances where a copy used\nanother copy's value (%)")
        if k == len(fams) - 1:   # labels in the last panel, where the right margin has room for them
            end_labels(ax, [(ob, "observed", c), (pr, "predicted", c)])
    save(fig, out, "fig1_cr")

    # 2: the missing value had been seen (meeting), observed against predicted
    fig, ax = plt.subplots(figsize=(a.width * 0.55, a.width * 0.4))
    for suf, fam in fams.items():
        c, m, _ = COND[suf]
        predicted(ax, pts(fam, "predicted", "found"), c, band=False)
        observed(ax, pts(fam, "observed", "found"), c, m)
    n_axis(ax, ns)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Chances where the missing\nvalue had been seen (%)")
    end_labels(ax, [(pts(f, "observed", "found") or pts(f, "predicted", "found"), COND[s][2], COND[s][0])
                    for s, f in fams.items()])
    save(fig, out, "fig2_found")

    # 3: twin effect, twin - aware text: observed paired by seed, and predicted from pairs
    pr_t = pred.get("twin_contrast", {}).get(f"{a.model} {a.prefix}{TWIN}", {})
    ob_t = []
    for n in ns:
        sel = [r for r in rows if r.get("n_agents") == n and r.get("awareness") == "aware"]
        pairs = pair_runs(sel, "twin", 0, 1)["pairs"]
        if len(pairs) >= 2:
            ob_t.append((n, *(100 * x for x in cluster_diff(pairs, cr))))
    if ob_t or pr_t:
        fig, ax = plt.subplots(figsize=(a.width * 0.5, a.width * 0.38))
        ax.axhline(0, color=GRAY, lw=0.7, zorder=1)
        c, m, _ = COND[TWIN]
        predicted(ax, sorted((int(n), *(100 * x for x in v)) for n, v in pr_t.items()), c)
        # N=2 hollow and unjoined: the told wording there ("Another instance", "A copy of you") differs from N>=4
        for sel, hollow in (([p for p in ob_t if p[0] == 2]), True), (([p for p in ob_t if p[0] > 2]), False):
            if sel:
                ax.errorbar([p[0] for p in sel], [p[1] for p in sel],
                            yerr=[[p[1] - p[2] for p in sel], [p[3] - p[1] for p in sel]], color=c, marker=m, ms=3.8,
                            mfc="white" if hollow else c, lw=0 if hollow else 1.3, elinewidth=0.8, capsize=0, zorder=3)
        n_axis(ax, ns)
        vals = [abs(v) for p in ob_t for v in p[2:]] + [abs(100 * v) for x in pr_t.values() for v in x[1:]]
        span = max(10.0, 1.15 * max(vals, default=0))
        ax.set_ylim(-span, span)
        ax.set_ylabel("Twin text − aware text\n(points)")
        save(fig, out, "fig3_twin")

    # 4: runs in which some copy used another's value
    fig, ax = plt.subplots(figsize=(a.width * 0.55, a.width * 0.4))
    for suf, fam in fams.items():
        c, m, _ = COND[suf]
        predicted(ax, pts(fam, "predicted", "p_any"), c, band=False)
        observed(ax, pts(fam, "observed", "p_any"), c, m)
    n_axis(ax, ns)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Runs where some copy used\nanother's value (%)")
    end_labels(ax, [(pts(f, "observed", "p_any") or pts(f, "predicted", "p_any"), COND[s][2], COND[s][0])
                    for s, f in fams.items()])
    save(fig, out, "fig4_pany")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
