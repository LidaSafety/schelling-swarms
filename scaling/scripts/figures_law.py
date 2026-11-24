#!/usr/bin/env python3
"""Figures 10-11: the title law and posting (scripts/title_law.py, scripts/analyze_posting.py). AAAI text width.

  .venv/bin/python scripts/figures_law.py results [--width 7.0]

fig10_title_law  collusion per chance against the copies holding each answer, this folder's runs only: measured
                 (95% intervals), the title law from the smallest runs (dashed, 95% band) and the one-number law
                 (dotted). a Qwen3-8B with 2 answers each, from pairs; b 1 answer each, from 4-copy runs, with the law
                 from 2-answer pairs for comparison; c Gemma 3 27B, from pairs.
fig11_posting    per model against swarm size: a holders posting their answer, b non-holders posting a wrong number
                 for the round's question, c found answers whose page also showed a wrong number. Qwen3-4B, 14B and
                 32B appear when analyze_posting.py was run with --extra ../wiki_game/data/swarm4round.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, GREEN, VERM, PURPLE, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7f7f7f"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})


def holders(x: dict) -> float:
    return sum(int(k) * v for k, v in x["holders"].items()) / sum(x["holders"].values())


def law_panel(ax, measured: list[dict], curve: list[dict], color: str, marker: str) -> None:
    cv = sorted(curve, key=holders)
    h = [holders(x) for x in cv]
    ax.fill_between(h, [100 * x["pred_cr_ci"][0] for x in cv], [100 * x["pred_cr_ci"][1] for x in cv],
                    color=GRAY, alpha=0.18, lw=0)
    ax.plot(h, [100 * x["pred_cr"] for x in cv], color="#333333", lw=1.2, ls="--")
    ax.plot(h, [100 * x["pred_cr_one"] for x in cv], color=GRAY, lw=1.0, ls=(0, (1, 1.5)))
    for x in measured:
        lo, hi = x["obs_cr_ci"]
        ax.errorbar(holders(x), 100 * x["obs_cr"], yerr=[[100 * (x["obs_cr"] - lo)], [100 * (hi - x["obs_cr"])]],
                    fmt=marker, color=color, ms=4.5, elinewidth=0.9, capsize=0, zorder=3)
    ax.set_xscale("log", base=2)
    hi = max(int(holders(x)) for x in cv)
    ticks = [2 ** k for k in range(hi.bit_length()) if 2 ** k <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Copies holding each answer")


def fig10(law: list[dict], out: Path, width: float) -> None:
    here = lambda fam, pre="N=": [x for x in law if x["family"].startswith(fam) and x["target"].startswith(pre)
                                  and "(" not in x["target"]]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(width, 0.336 * width), sharey=True, gridspec_kw={"wspace": 0.12})
    two = here("Qwen3-8B, 2 answers each")
    law_panel(a, two, two, BLUE, "o")
    a.set_ylabel("Collusion per chance (%)")
    a.text(0.0, 1.03, "a  Qwen3-8B, 2 answers each", transform=a.transAxes, va="bottom", fontsize=8)
    a.text(4.6, 84, "one\nnumber", fontsize=6.5, color=GRAY, ha="right", va="center")
    a.text(14, 47, "title law\nfrom pairs", fontsize=6.5, color="#333333", ha="left", va="top")
    one = here("Qwen3-8B, 1 answer each")
    law_panel(b, one, one, ORANGE, "D")
    cross = sorted(here("Qwen3-8B, 2 answers each", "1 answer each N="), key=holders)
    b.plot([holders(x) for x in cross], [100 * x["pred_cr"] for x in cross], color=BLUE, lw=1.0, ls="--", alpha=0.8)
    b.text(32, 58, "law from\n2-answer pairs", fontsize=6.5, color=BLUE, ha="right", va="top")
    b.text(0.0, 1.03, "b  Qwen3-8B, 1 answer each", transform=b.transAxes, va="bottom", fontsize=8)
    gem = here("Gemma 3 27B")
    law_panel(c, gem, gem, GREEN, "^")
    c.text(0.0, 1.03, "c  Gemma 3 27B, 2 answers each", transform=c.transAxes, va="bottom", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig10_title_law.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig11(post: dict, out: Path, width: float) -> None:
    models = [("Qwen3-4B", VERM, "v"), ("Qwen3-8B", BLUE, "o"), ("Qwen3-14B", PURPLE, "s"), ("Qwen3-32B", GRAY, "P"),
              ("Gemma 3 27B", GREEN, "^")]
    series = {}
    for name, *_ in models:
        by_n = {}
        for x in post["cells"]:
            if x["family"] == name and (x["N"] not in by_n or x["source"] == "results"):
                by_n[x["N"]] = x
        if by_n:
            series[name] = [by_n[n] for n in sorted(by_n)]
    fig, axs = plt.subplots(1, 3, figsize=(width, 0.336 * width), gridspec_kw={"wspace": 0.62})
    keys = [("posted", "Holders posting the answer (%)"), ("wrong_notes", "Non-holders posting a wrong one (%)"),
            ("beside_wrong", "Found beside a wrong one (%)")]
    ns = sorted({x["N"] for s in series.values() for x in s})
    for ax, (k, lab), letter in zip(axs, keys, "abc"):
        for name, col, mk in models:
            d = [x for x in series.get(name, []) if x[k] is not None]
            if d:
                ax.plot([x["N"] for x in d], [100 * x[k] for x in d], marker=mk, color=col, ms=3.5, lw=1.1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) if n in (2, 8, 32, 128) or len(ns) <= 5 else "" for n in ns])
        ax.set_xlabel("Copies")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 102)
        ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    for name, col, mk in models:
        if name in series:
            last = [x for x in series[name] if x["posted"] is not None][-1]
            dy = {"Qwen3-4B": 3, "Qwen3-8B": -2}.get(name, 0)
            axs[0].text(last["N"] * 1.2, 100 * last["posted"] + dy, name, color=col, fontsize=6.5, va="center")
    axs[0].set_xlim(ns[0] / 1.2, ns[-1] * 4.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig11_posting.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--width", type=float, default=7.0, help="inches (AAAI text width 7.0)")
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "figures"
    out.mkdir(exist_ok=True)
    fig10(json.loads((res / "title_law.json").read_text()), out, a.width)
    fig11(json.loads((res / "posting.json").read_text()), out, a.width)
    print("wrote", out / "fig10_title_law.png", "and", out / "fig11_posting.png")


if __name__ == "__main__":
    main()
