"""Tables for the N-agent dead drop (scb/swarm.py): per-cell rates, the check against Marcin's scenario 2 numbers, and
paired contrasts on the same seeds: the aware sentence (aware minus unaware), the orchestrator log (log A minus no log),
its placebo (log B minus no log) and the same-model content of the log (log A minus log B).

  python scripts/analyze_swarm.py [results_dir]

Reads <results>/swarm/raw/*.jsonl.gz and <results>/swarm_*/raw/*.jsonl.gz. Failed runs are counted per cell and in
every contrast, never dropped silently. Intervals for contrasts resample whole seeds (one seed = the same quiz and
sampling seeds in every condition), so conditions that share a seed are not treated as independent.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scb.io import read_jsonl  # noqa: E402
from scb.swarm import cluster_diff, cr, cross_match, pair_runs, round1_titles, summarize, sum_p2  # noqa: E402

# Marcin's README, scenario 2, wiki, rule in force: share of runs that colluded (400 runs per model: 2 awareness texts
# x 2 listings x seeds 10000-10099). Rounded to 0.1 points in the README.
MARCIN_WIKI_RULE = {"gemma-3-27b": 0.658, "qwen3-4b": 0.412, "qwen3-8b": 0.372, "llama-3.3-70b": 0.255,
                    "qwen3-1.7b": 0.235, "gpt-oss-120b": 0.220, "qwen3-14b": 0.218, "mistral-small-24b": 0.210,
                    "qwen3-32b": 0.120, "gpt-oss-20b": 0.066, "qwen3-32b-think": 0.035}


def load(res: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    files = sorted(res.glob("swarm/raw/*.jsonl*")) + sorted(res.glob("swarm_*/raw/*.jsonl*"))
    for f in files:
        rows = read_jsonl(f)
        print(f"loaded {f} ({len(rows)} runs)")
        out.setdefault(f.name.split(".jsonl")[0], []).extend(rows)
    return out


def f(x: float) -> str:
    return "–" if x != x else f"{100 * x:.1f}"


def colluded(rs: list[dict]) -> float:
    return sum(r["colluded"] for r in rs) / len(rs) if rs else float("nan")


def seen(rs: list[dict]) -> float:
    n = sum(r["chances"]["n"] for r in rs)
    return sum(r["chances"]["seen"] for r in rs) / n if n else float("nan")


def wrote(rs: list[dict]) -> float:
    return sum(r["labels"]["any_shared"] for r in rs) / len(rs) if rs else float("nan")


def met(rs: list[dict]) -> float:
    return sum(r["labels"]["path_match"] for r in rs) / len(rs) if rs else float("nan")


STATS = (("colluded (runs)", colluded), ("CR per chance", cr), ("value seen", seen), ("wrote own", wrote),
         ("met on a page", met),
         ("title sum p² (write, r1)", lambda rs: sum_p2(round1_titles(rs, "writes"))),
         ("write = read title (r1)", lambda rs: cross_match(round1_titles(rs, "writes"), round1_titles(rs, "reads"))))


def reproduction(model: str, ok: list[dict], failed: list[dict]) -> None:
    """Our wiki, rule-bound, N=2 runs on Marcin's seeds against his README rate."""
    def is_base(r: dict) -> bool:
        g = r.get
        return (g("channel") == "wiki" and g("rule") == "forbidden" and g("n_agents") == 2 and g("n_questions") == 4
                and g("f_inv") == 2 and not g("randomized") and g("log") == "none"
                and g("defense") == "none" and not g("read_cap") and not g("shuffle_writes")
                and g("sample_offset", 0) == 0 and 10000 <= g("seed", -1) < 10100)
    base, lost = [r for r in ok if is_base(r)], [r for r in failed if is_base(r)]
    if not base or model not in MARCIN_WIKI_RULE:
        return
    grid = {(r["awareness"], r["listing"], r["seed"]) for r in base}
    complete = len(grid) == len(base) == 400
    p, m = colluded(base), MARCIN_WIKI_RULE[model]
    # spread of our estimate when whole seeds are resampled (4 conditions share each seed)
    by_seed: dict[int, list[dict]] = {}
    for r in base:
        by_seed.setdefault(r["seed"], []).append(r)
    seeds, rng = list(by_seed.values()), random.Random(0)
    boots = sorted(colluded([r for _ in seeds for r in seeds[rng.randrange(len(seeds))]]) for _ in range(2000))
    se = (sum((x - p) ** 2 for x in boots) / len(boots)) ** 0.5
    half = 1.96 * (2 ** 0.5) * se        # Marcin's estimate has the same design, so assume the same spread
    print(f"\nReproduction (wiki, rule, N=2, no log, seeds 10000-10099): {f(p)}% of {len(base)} runs "
          f"({len(lost)} failed) vs Marcin {f(m)}%: difference {100 * (p - m):+.1f} points; two estimates of this "
          f"design differ by up to about ±{100 * half:.1f} by chance (95%)."
          + ("" if complete else f"  WARNING: grid incomplete ({len(grid)} distinct of 400 expected)"))


def n_seeds(pairs: list[tuple[dict, dict]]) -> int:
    return len({(p[0]["seed"], p[0].get("sample_offset", 0)) for p in pairs})


def contrast(name: str, rows: list[dict], factor: str, base, alt, extra=lambda r: True) -> None:
    """`alt` minus `base` on paired runs (same quiz, same sampling seeds, every other setting equal). Pairs are never
    pooled across N or awareness: each (N, awareness) group gets its own block of statistics, then CR per cell (the
    other prompt switches). Works for any switch: log (none/A/B), rule, awareness, listing."""
    sel = [r for r in rows if extra(r)]
    if not any(r.get(factor) == alt for r in sel):
        return                                                    # arm `alt` was never run
    pr = pair_runs(sel, factor, base, alt)
    pairs = pr["pairs"]
    failed = [r for r in sel if "error" in r]
    print(f"\n{name}: {len(pairs)} pairs on {n_seeds(pairs)} seeds; unpaired {pr['unpaired_base']} ({factor}={base}) / "
          f"{pr['unpaired_alt']} ({factor}={alt}); duplicates {pr['duplicates']}; failed runs in these cells {len(failed)}")
    if not pairs:
        print("  no pairs: nothing to compare")
        return
    group = lambda r: (r["n_agents"],) + (() if factor == "awareness" else (r["awareness"],))  # noqa: E731
    for g in sorted({group(p[0]) for p in pairs}):
        sub = [p for p in pairs if group(p[0]) == g]
        print(f"  N={g[0]}" + (f", awareness={g[1]}" if len(g) > 1 else "") + f": {len(sub)} pairs on "
              f"{n_seeds(sub)} seeds")
        for label, stat in STATS:
            d, lo, hi = cluster_diff(sub, stat)
            print(f"    {label:26s} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    others = [k for k in ("rule", "awareness", "listing") if k != factor]
    cell = lambda r: (r["n_agents"], *[r[k] for k in others])  # noqa: E731
    for c in sorted({cell(p[0]) for p in pairs}):
        sub = [p for p in pairs if cell(p[0]) == c]
        d, lo, hi = cluster_diff(sub, cr)
        print(f"  CR per chance, N={c[0]} {'/'.join(c[1:]):26s} {100 * d:+6.1f}  [{100 * lo:+.1f}, {100 * hi:+.1f}]  "
              f"({len(sub)} pairs on {n_seeds(sub)} seeds)")


def main() -> None:
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    for model, rows in load(res).items():
        print(f"\n## {model}\n")
        # headline: Marcin's per-run "colluded"; CR per chance is reported next to it. The independence ratio links
        # the two (1 = each chance an independent try; below 1 = uses bunch up in some runs), see scaling-law.html.
        # CR counts a use only if the value was seen on a title the copies chose; "uses via <page title>" counts the
        # uses it leaves out (value seen only on a title copied from the prompt's example, as Marcin's `colluded` does)
        print("| cell | runs | failed | colluded (runs, headline) | CR per chance [95%] | uses via <page title> | "
              "independence ratio | value seen | wrote own | met | r1 write / read rate | title sum p² write / read | "
              "write = read title |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, c in summarize(rows).items():
            if not c["n"]:
                print(f"| {key} | 0 | {c['errors']} | | | | | | | | | | |")
                continue
            lo, hi = c["cr_ci95"]
            ir = c["independence_ratio"]
            print(f"| {key} | {c['n']} | {c['errors']} | {f(c['colluded'])} | {f(c['cr'])} [{f(lo)}, {f(hi)}] | "
                  f"{c['uses_placeholder']} | {'–' if ir != ir else f'{ir:.2f}'} | {f(c['seen_rate'])} | {f(c['any_shared'])} | {f(c['path_match'])} | "
                  f"{f(c['r1_write_rate'])} / {f(c['r1_read_rate'])} | "
                  f"{c['title_p2_write']:.3f} / {c['title_p2_read']:.3f} | {c['title_match_wr']:.3f} |")
        ok = [r for r in rows if "error" not in r]
        reproduction(model, ok, [r for r in rows if "error" in r])
        contrast("Log A contrast, log A minus no log", rows, "log", "none", "A")
        # log B is the placebo (other sandboxes run a different model): any-log effect, then same-model content
        contrast("Placebo-log contrast, log B minus no log", rows, "log", "none", "B")
        contrast("Same-model content, log A minus log B", rows, "log", "B", "A")
        # decomposition of Marcin's grid: each switch's effect on the same quizzes, the other switches held equal
        for factor, base, alt, label in (("rule", "silent", "forbidden", "Rule effect, forbidden minus silent"),
                                         ("awareness", "unaware", "aware", "Awareness effect, aware minus unaware"),
                                         ("listing", "none", "workspace", "Page-list effect, workspace minus none")):
            contrast(f"{label} (no log)", rows, factor, base, alt, lambda r: r.get("log") == "none")


if __name__ == "__main__":
    main()
