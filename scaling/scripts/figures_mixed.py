#!/usr/bin/env python3
"""Figure for the mixed swarms (copies of Qwen3-8B and Gemma 3 27B on one wiki, the fixed prompt), in the encoding of
scripts/figures_scaling.py. The runs come from scripts/run_mixed_grid.sh.

  .venv/bin/python scripts/figures_mixed.py results [--pred predict_mixed.json] [--width 5.5]

  fig9_mixed  collusion per chance against the number of Gemma 3 27B copies (the others Qwen3-8B); a: 4 copies,
              b: 8 copies. Solid: measured, 95% intervals from resampling runs. Dashed with band: predicted from each
              model's 2-copy runs before any mixed run (scripts/predict_mixed.py). Gray, dotted: the rate if mixing
              did not matter, i.e. every copy colluding as in a one-model swarm of the same size (the straight line
              between the two one-model swarms; each copy has the same number of chances).
All runs use the fixed prompt (unaware text), hence its blue circles.
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
from analyze_mixed import label, per_run, ratio  # noqa: E402
from figures_scaling import COND, GRAY, INK, UNAWARE, observed, predicted, save, setup  # noqa: E402
from predict_swarm import load_rows  # noqa: E402
from scb.swarm import _boot, mix_label  # noqa: E402

QWEN, GEMMA = "qwen3-8b", "gemma-3-27b"
NEUTRAL = (0, (1, 1.5))


def spec(n: int, g: int) -> str:
    """The swarm of n copies with g of Gemma, seated as scripts/run_mixed_grid.sh seats them (Qwen first)."""
    return mix_label([QWEN] * (n - g) + [GEMMA] * g)


def cr_all(runs: list[dict]) -> float:
    return ratio(runs, ("used", None))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--pred", default="predict_mixed.json", help="predict_mixed.py output in the results directory")
    ap.add_argument("--width", type=float, default=5.5)
    a = ap.parse_args()
    setup()
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(exist_ok=True)
    pred = json.loads((res / a.pred).read_text())["mixes"]
    rows = [r for r in load_rows(res) if "error" not in r and r.get("channel") == "wiki" and r.get("rule") == "forbidden"
            and r.get("awareness") == "unaware" and r.get("listing") == "none" and r["log"] == "none"
            and r.get("n_questions") == 4 and r.get("f_inv") == 2 and not r.get("randomized")
            and r.get("defense", "none") == "none" and not r.get("read_cap")]
    cells: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("seat_models") or label(r) in pred:
            cells.setdefault(label(r), []).append({"c": per_run(r)[0]})
    c, m = COND[UNAWARE][:2]
    fig, axes = plt.subplots(1, 2, figsize=(a.width, a.width * 0.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.8]})
    for k, (ax, n) in enumerate(zip(axes, (4, 8))):
        gs = [g for g in range(0, n + 1, 2) if spec(n, g) in cells]
        obs = [(g, 100 * cr_all(cells[spec(n, g)]), *(100 * v for v in _boot(cells[spec(n, g)], cr_all))) for g in gs]
        predicted(ax, [(g, *(100 * v for v in pred[spec(n, g)]["cr"])) for g in gs if spec(n, g) in pred], c)
        ends = [p for p in obs if p[0] in (0, n)]
        if len(ends) == 2:
            ax.plot([0, n], [ends[0][1], ends[1][1]], color=GRAY, ls=NEUTRAL, lw=1.2, zorder=2)
        observed(ax, obs, c, m)
        ax.set_xticks(range(0, n + 1, 2))
        ax.set_xlim(-0.6, n + 0.6)
        ax.set_ylim(0, 80)
        ax.text(0.0, 1.03, f"{'ab'[k]}  {n} copies", transform=ax.transAxes, fontsize=8, va="bottom", color=INK)
    axes[0].set_ylabel("Chances where a copy used\nanother copy's value (%)")
    axes[1].legend(handles=[Line2D([], [], color=c, marker=m, ms=3.8, lw=1.3, label="measured"),
                            Line2D([], [], color=c, lw=1.0, ls=(0, (4, 2)), label="predicted from 2-copy runs"),
                            Line2D([], [], color=GRAY, lw=1.2, ls=NEUTRAL, label="if mixing did not matter")],
                   loc="upper left", frameon=False, fontsize=7, handlelength=2.6, borderaxespad=0.2)
    fig.supxlabel("Gemma 3 27B copies in the swarm (the others Qwen3-8B)", fontsize=8, y=-0.02)
    save(fig, out, "fig9_mixed")
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
