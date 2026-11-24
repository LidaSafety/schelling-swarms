#!/usr/bin/env python3
"""Figures for the design-knob runs off the fixed prompt (unaware, no log, rule in force, no listing; Qwen3-8B), in
the encoding of scripts/figures_scaling.py. The runs come from scripts/run_branch_grid.sh.

  .venv/bin/python scripts/predict_swarm.py results --aware-base 4 --out predict_swarm_all.json
  .venv/bin/python scripts/figures_knobs.py results --pred predict_swarm_all.json [--width 5.5]

  fig7_rounds  Q=8 against Q=4 questions (f_inv=2, fixed blocks). a: collusion per chance, measured (solid, 95%
               intervals) and predicted from pairs (dashed). b: runs in which some copy used another's value, with
               the rate 8 rounds would give if they were two independent 4-round games, 1 - (1 - rate at Q=4)^2
               (gray, dotted).
  fig8_split   random splits (R=1: each copy holds its own random Q/f_inv questions) against fixed blocks (R=0), Q=4,
               f_inv=2. a: collusion per chance. b: R=1 - R=0, paired by seed (same quiz and true values).
The baseline (Q=4, f_inv=2, R=0) keeps the blue circles of the unaware text; Q=8 is reddish purple diamonds and R=1
orange down-triangles, colors the told conditions do not use. Two series at one N are offset a little sideways.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_scaling import COND, GRAY, INK, UNAWARE, n_axis, predicted, save, setup  # noqa: E402
from predict_swarm import load_rows  # noqa: E402
from scb.swarm import _boot, cluster_diff, colluded, cr, pair_runs  # noqa: E402

NS = [2, 4, 8, 16, 32]
BASE_C, BASE_M = COND[UNAWARE][:2]
Q8_C, Q8_M = "#CC79A7", "D"
R1_C, R1_M = "#E69F00", "v"
LEFT, RIGHT = 0.94, 1.06          # sideways offsets (factors on the log axis) for two series at one N
CRV2 = "Chances where a copy used\nanother copy's value (%)"
CRV1 = "Runs where some copy used\nanother's value (%)"


def any_use(g: list[dict]) -> float:
    return sum(colluded(r) for r in g) / len(g)


def series(cells: dict, key: tuple, stat, dx: float = 1.0) -> list[tuple]:
    """[(N * dx, value, lo, hi)] in percent, for the cells (Q, f_inv, R, N) of `key` that have runs."""
    out = []
    for n in NS:
        g = cells.get((*key, n))
        if g:
            lo, hi = _boot(g, stat)
            out.append((n * dx, 100 * stat(g), 100 * lo, 100 * hi))
    return out


def bars(ax, pts: list[tuple], color: str, marker: str) -> None:
    if pts:
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                    color=color, marker=marker, ms=3.8, lw=1.3, elinewidth=0.8, capsize=0, zorder=3)


def handle(color: str, marker: str | None, label: str, ls="-") -> Line2D:
    return Line2D([], [], color=color, marker=marker, ms=3.8, lw=1.3 if ls == "-" else 1.0, ls=ls, label=label)


def panel(ax, k: int, label: str, ylabel: str) -> None:
    ax.text(0.0, 1.03, f"{'ab'[k]}  {label}", transform=ax.transAxes, fontsize=8, va="bottom", color=INK)
    ax.set_ylabel(ylabel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--pred", default="predict_swarm.json", help="predict_swarm.py output inside the results directory")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--width", type=float, default=5.5)
    a = ap.parse_args()
    setup()
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(exist_ok=True)
    fams = json.loads((res / a.pred).read_text())["families"]
    rows = [r for r in load_rows(res) if "error" not in r and r.get("model") == a.model and r.get("channel") == "wiki"
            and r.get("rule") == "forbidden" and r.get("listing") == "none" and r["log"] == "none"
            and r.get("awareness") == "unaware" and r.get("defense", "none") == "none" and not r.get("read_cap")]
    cells: dict[tuple, list[dict]] = {}
    for r in rows:
        cells.setdefault((r["n_questions"], r["f_inv"], int(r["randomized"]), r["n_agents"]), []).append(r)

    # 7: more rounds. a: per chance, measured and predicted; b: runs with any use, against independent rounds
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(a.width, a.width * 0.4))
    for q, c, m, dx in ((4, BASE_C, BASE_M, LEFT), (8, Q8_C, Q8_M, RIGHT)):
        fam = fams.get(f"{a.model} wiki/forbidden/unaware/none/N*-Q{q}-finv2-R0-log-none", {})
        predicted(a1, sorted((int(n), *(100 * x for x in v["predicted"]["cr"])) for n, v in fam.items()), c, band=False)
        bars(a1, series(cells, (q, 2, 0), cr, dx), c, m)
        bars(a2, series(cells, (q, 2, 0), any_use, dx), c, m)
    base = series(cells, (4, 2, 0), any_use)
    a2.plot([p[0] for p in base], [100 * (1 - (1 - p[1] / 100) ** 2) for p in base], color=GRAY, ls=(0, (1, 1.5)),
            lw=1.2, zorder=2)
    for k, (ax, label, ylabel) in enumerate(((a1, "CR-v2, per chance", CRV2), (a2, "CR-v1, per run", CRV1))):
        n_axis(ax, NS)
        ax.set_ylim(0, 102)
        panel(ax, k, label, ylabel)
    a1.legend(handles=[handle(GRAY, "o", "measured"), handle(GRAY, None, "predicted from pairs", (0, (4, 2)))],
              loc="upper left", frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    a2.legend(handles=[handle(BASE_C, BASE_M, "4 questions"), handle(Q8_C, Q8_M, "8 questions"),
                       handle(GRAY, None, "8, if rounds were\nindependent", (0, (1, 1.5)))],
              loc="lower right", frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.45)
    save(fig, out, "fig7_rounds")

    # 8: random splits. a: per chance; b: R=1 - R=0 paired by seed
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(a.width, a.width * 0.4))
    bars(a1, series(cells, (4, 2, 0), cr, LEFT), BASE_C, BASE_M)
    bars(a1, series(cells, (4, 2, 1), cr, RIGHT), R1_C, R1_M)
    diff = []
    for n in NS:
        sel = [r for r in rows if r["n_questions"] == 4 and r["f_inv"] == 2 and r["n_agents"] == n]
        pairs = pair_runs(sel, "randomized", 0, 1)["pairs"]
        if len(pairs) >= 2:
            diff.append((n, *(100 * x for x in cluster_diff(pairs, cr))))
    a2.axhline(0, color=GRAY, lw=0.7, zorder=1)
    bars(a2, diff, R1_C, R1_M)
    span = max(10.0, 1.15 * max((abs(v) for p in diff for v in p[2:]), default=0))
    for k, (ax, label, ylabel) in enumerate(((a1, "CR-v2, per chance", CRV2),
                                             (a2, "paired by seed", "Random split − fixed\nblocks (points)"))):
        n_axis(ax, NS)
        panel(ax, k, label, ylabel)
    a1.set_ylim(0, 102)
    a2.set_ylim(-span, span)
    a1.legend(handles=[handle(BASE_C, BASE_M, "fixed blocks"), handle(R1_C, R1_M, "random split")],
              loc="upper left", frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.45)
    save(fig, out, "fig8_split")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
