#!/usr/bin/env python3
"""Figures for the two extra experiments, in the encoding of scripts/figures_scaling.py.

  .venv/bin/python scripts/figures_extra.py results [--width 5.5]

  fig5_warning      change in collusion per chance when the benchmark's warning note (scb/defense.py "warning") is
                    added, at each N, paired by seed with the runs without it; one series per condition
  fig6_composition  collusion per chance against the number of copies holding each answer (N / f_inv): copies holding
                    2 of the 4 answers (f_inv=2, solid, filled) or 1 (f_inv=4, dashed, hollow). If only the holders
                    mattered the two lines would coincide; a gap means the swarm's total size matters too.
Hollow markers at N=2 in the told conditions: their wording there ("Another instance", "A copy of you") differs from
N>=4, as in the other figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_scaling import AWARE, COND, GRAY, INK, TWIN, UNAWARE, end_labels, save, setup  # noqa: E402
from predict_swarm import load_rows  # noqa: E402
from scb.swarm import _boot, cluster_diff, cr, pair_runs  # noqa: E402

KEYS = [("unaware", 0, UNAWARE), ("aware", 0, AWARE), ("aware", 1, TWIN)]   # (awareness, twin, family suffix)


def log2_axis(ax, ticks: list[int], label: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(ticks[0] / 1.25, ticks[-1] * 1.25)
    ax.set_xlabel(label)


def bars(ax, pts: list[tuple], color: str, marker: str, hollow: bool, ls: str = "-") -> None:
    if pts:
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    color=color, marker=marker, mfc="white" if hollow else color, ms=3.8, lw=1.3, ls=ls,
                    elinewidth=0.8, capsize=0, zorder=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--width", type=float, default=5.5)
    a = ap.parse_args()
    setup()
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(exist_ok=True)
    rows = [r for t in ("main_n*", "comp_n*", "def_n*") for r in load_rows(res, f"swarm_{t}/raw/*.jsonl*")
            if "error" not in r and r.get("model") == a.model and r.get("channel") == "wiki"
            and r.get("rule") == "forbidden" and r.get("listing") == "none" and not r.get("randomized")
            and r["log"] == "none" and r.get("n_questions") == 4 and not r.get("read_cap")]

    # 5: the warning note's effect, paired by seed with the same cell without it
    fig, ax = plt.subplots(figsize=(a.width * 0.62, a.width * 0.42))
    ax.axhline(0, color=GRAY, lw=0.7, zorder=1)
    labels = []
    for aw, tw, suf in KEYS:
        c, m, lab = COND[suf]
        pts = []
        for n in (2, 4, 8, 16):
            sel = [r for r in rows if r["n_agents"] == n and r["f_inv"] == 2 and r["awareness"] == aw and r["twin"] == tw]
            pairs = pair_runs(sel, "defense", "none", "warning")["pairs"]
            if len(pairs) >= 2:
                pts.append((n, *(100 * x for x in cluster_diff(pairs, cr))))
        told = aw == "aware"
        bars(ax, [p for p in pts if p[0] > 2 or not told], c, m, hollow=False)
        bars(ax, [p for p in pts if p[0] == 2 and told], c, m, hollow=True, ls="none")
        labels.append((pts, lab, c))
    log2_axis(ax, [2, 4, 8, 16], "Copies in the swarm (N)")
    ax.set_ylim(-55, 30)
    ax.set_ylabel("Change in collusion with\nthe warning note (points)")
    end_labels(ax, labels)
    save(fig, out, "fig5_warning")

    # 6: composition, against copies holding each answer
    fig, axes = plt.subplots(1, 3, figsize=(a.width, a.width * 0.36), sharey=True)
    for k, (ax, (aw, tw, suf)) in enumerate(zip(axes, KEYS)):
        c, m, lab = COND[suf]
        told = aw == "aware"
        series = []
        for f_inv, ls, hollow, name in ((2, "-", False, "2 answers each"), (4, (0, (4, 2)), True, "1 answer each")):
            pts = []
            for n in sorted({r["n_agents"] for r in rows if r["f_inv"] == f_inv}):
                g = [r for r in rows if r["n_agents"] == n and r["f_inv"] == f_inv and r["awareness"] == aw
                     and r["twin"] == tw and r.get("defense", "none") == "none"]
                if g:
                    lo, hi = _boot(g, cr)
                    pts.append((n // f_inv, 100 * cr(g), 100 * lo, 100 * hi, n))
            main = [p for p in pts if not (told and p[4] == 2)]
            bars(ax, [p[:4] for p in main], c, m, hollow=hollow, ls=ls)
            if told:
                bars(ax, [p[:4] for p in pts if p[4] == 2], c, m, hollow=True, ls="none")
            series.append(([p[:4] for p in main], name, c))
        log2_axis(ax, [1, 2, 4, 8, 16], "")
        ax.set_ylim(0, 100)
        ax.text(0.0, 1.03, f"{'abc'[k]}  {lab}", transform=ax.transAxes, fontsize=8, va="bottom", color=INK)
        if k == 0:
            ax.set_ylabel("Chances where a copy used\nanother copy's value (%)")
        if k == 0:   # one key for the line encoding, in the empty upper-left corner (end labels collide here)
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color=GRAY, marker="o", ms=3.8, lw=1.3, label="2 answers each"),
                               Line2D([], [], color=GRAY, marker="o", mfc="white", ms=3.8, lw=1.3, ls=(0, (4, 2)),
                                      label="1 answer each")],
                      loc="upper left", frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    fig.supxlabel("Copies holding each answer", fontsize=8, y=-0.02)
    save(fig, out, "fig6_composition")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
