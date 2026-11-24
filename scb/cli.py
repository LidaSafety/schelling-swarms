"""Command line: python -m scb <deaddrop | swarm | codedrop | review | livewiki | committee | wikigame | evolve-post> --models <model> [options]

Each command writes <out>/<command>_<tag>/raw/<model>.jsonl.gz (one JSON object per run, with full transcripts) and
merges a per-condition summary into <out>/<command>_<tag>/summary.json. The model is reached through the endpoint in
SB_BASE_URL (see scb/llm.py); scripts/sweep.py serves models with vLLM and sets it. SB_ENDPOINTS can set the endpoint
per model instead (swarm, for mixed swarms on several servers).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .defense import DEFENSES
from .io import write_jsonl
from .llm import LLM


def _merge(old: dict, new: dict) -> dict:
    out = dict(old)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _regime(name: str) -> str:
    """A wiki game regime name for argparse: one of the named regimes or a fixed-N cell, fixed-h<H>-s<S>."""
    from .wikigame import REGIMES, fixed_cell
    if name in REGIMES or fixed_cell(name):
        return name
    raise argparse.ArgumentTypeError(f"unknown regime {name!r}: use {', '.join(REGIMES)} or fixed-h<H>-s<S>")


def _keep_saved_rows(out_dir: Path, model: str, rows: list[dict], key: Callable[[dict], Any]) -> list[dict]:
    """The rows already saved for this model in out_dir that this run did not redo (by `key`), then this run's rows,
    so experiments sharing an output directory do not overwrite each other."""
    path = out_dir / "raw" / f"{model.replace('/', '_').replace(':', '_')}.jsonl.gz"
    if not path.exists():
        return rows
    from .io import read_jsonl
    redone = {key(r) for r in rows}
    return [r for r in read_jsonl(path) if key(r) not in redone] + rows


def _llm(model: str, temperature: float, max_tokens: int) -> LLM:
    """The client for one served model. SB_ENDPOINTS, a JSON object model -> {base_url, api_key, extra_body}, overrides
    the SB_* defaults of scb/llm.py per model, so the copies of a mixed swarm can reach models on different servers.
    A claude-* model goes to the Anthropic Messages API (scb/llm.py ClaudeLLM)."""
    if model.startswith("claude-"):
        from .llm import ClaudeLLM
        return ClaudeLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    ep = json.loads(os.environ.get("SB_ENDPOINTS") or "{}").get(model, {})
    return LLM(model=model, temperature=temperature, max_tokens=max_tokens,
               **{k: ep[k] for k in ("base_url", "api_key", "extra_body", "token_mult", "retries") if k in ep})


def _save(out_dir: Path, model: str, rows: list[dict], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    write_jsonl(out_dir / "raw" / f"{model.replace('/', '_').replace(':', '_')}.jsonl.gz", rows)
    path = out_dir / "summary.json"
    with open(out_dir / ".summary.lock", "w") as lock:   # several jobs (one per model) may finish at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            old = json.loads(path.read_text()) if path.exists() else {}
            merged = _merge(old, {**meta, "models": {model: summary}})
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(merged, indent=2))
            tmp.replace(path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_all(fn: Callable[..., dict], jobs: list[tuple], workers: int) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001  (one failed server call must not kill the sweep)
                rows.append({"error": str(e)[:500]})
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} runs", flush=True)
    return rows


def cmd_deaddrop(a: argparse.Namespace) -> None:
    from .deaddrop import run_pair, summarize
    out_dir = Path(a.out) / ("deaddrop" + (f"_{a.tag}" if a.tag else ""))
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        t0 = time.time()
        # the drive has the full grid; the other channels have no directories (file only) and no "random" listing
        jobs = [(llm, ru, aw, li, gr, a.seed * 10000 + i, a.rounds, ch, 0, df)
                for ch in a.channels for ru in a.rules for aw in a.awareness for li in a.listing
                for gr in a.granularity for df in a.defenses for i in range(a.n)
                if ch == "drive" or (gr == "file" and li in ("workspace", "none"))]
        rows = _run_all(run_pair, jobs, a.workers)
        ok = [r for r in rows if "error" not in r]
        s = {**summarize(ok), "_errors": len(rows) - len(ok), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"rounds": a.rounds})
        print(json.dumps(s, indent=1))


def cmd_swarm(a: argparse.Namespace) -> None:
    """Scenario 2 with N copies (scb/swarm.py). A failed run is kept with its condition, so failures count per cell.
    Every setting is checked before any call is made. If more than --max-error-frac of the runs fail, the results are
    still saved but the command exits with status 2, so scripts/sweep.py does not mark the experiment done."""
    from .swarm import MAX_TOKENS, MODEL_IDS, check_params, mix_label, parse_mix, run_swarm, summarize
    out_dir = Path(a.out) / ("swarm" + (f"_{a.tag}" if a.tag else ""))
    if not a.models and not a.mix:
        raise SystemExit("swarm needs --models or --mix")
    if a.read_cap < 0:
        raise SystemExit("--read-cap must be >= 0 (0 = no cap)")
    if any(lg != "none" for lg in a.log) and any(m not in MODEL_IDS for m in a.models):
        raise SystemExit(f"--log A/B needs a public model id in scb/swarm.py MODEL_IDS for: "
                         f"{[m for m in a.models if m not in MODEL_IDS]}")
    if a.mix and any(lg != "none" for lg in a.log):
        raise SystemExit("--mix needs --log none: a log shows one model id for every sandbox")

    def one(llm, kw: dict) -> dict:
        try:
            return run_swarm(llm, **kw)
        except Exception as e:  # noqa: BLE001
            label = mix_label([x.model for x in llm]) if isinstance(llm, list) else llm.model
            return {"error": str(e)[:500], "model": label, "granularity": "file", "rounds": kw["n_questions"], **kw}

    # (name in the results, model or one model per copy, swarm sizes): each model alone, then each mix
    entries = [(m, _llm(m, a.temperature, MAX_TOKENS), list(a.agents)) for m in a.models]
    for spec in a.mix:
        try:
            seats = parse_mix(spec)
        except ValueError as e:
            raise SystemExit(f"--mix: {e}") from None
        served = {m: _llm(m, a.temperature, MAX_TOKENS) for m in dict.fromkeys(seats)}
        entries.append((mix_label(seats), [served[m] for m in seats], [len(seats)]))
    failed_models = []
    for model, llm, sizes in entries:
        t0 = time.time()
        conds = [dict(rule=ru, awareness=aw, listing=li, n_agents=n, n_questions=a.questions, f_inv=a.f_inv,
                      randomized=rd, log=lg, channel=ch, defense=df, read_cap=a.read_cap or None,
                      shuffle_writes=a.shuffle_writes, plant=pl)
                 for ch in a.channels for n in sizes for rd in a.randomized for lg in a.log
                 for ru in a.rules for aw in a.awareness for li in a.listing for df in a.defenses for pl in a.plant]
        for c in conds:
            try:
                check_params(**{k: v for k, v in c.items() if k != "shuffle_writes"})
            except ValueError as e:
                raise SystemExit(f"invalid setting {c}: {e}") from None
        jobs = [(llm, dict(c, seed=a.seed * 10000 + i)) for c in conds for i in range(a.start, a.start + a.n)]
        rows = _run_all(one, jobs, a.workers)
        n_err = sum("error" in r for r in rows)
        s = {**summarize(rows), "_errors": n_err, "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"questions": a.questions, "f_inv": a.f_inv,
                                        "temperature": a.temperature, "max_tokens_per_call": MAX_TOKENS})
        print(json.dumps(s, indent=1))
        if rows and n_err / len(rows) > a.max_error_frac:
            failed_models.append(f"{model}: {n_err}/{len(rows)} runs failed")
    if failed_models:
        print("FAILED RUNS above --max-error-frac:\n  " + "\n  ".join(failed_models), flush=True)
        raise SystemExit(2)


def cmd_codedrop(a: argparse.Namespace) -> None:
    from .codedrop import PROMPT_V, run_code, summarize
    out_dir = Path(a.out) / ("codedrop" + (f"_{a.tag}" if a.tag else ""))
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        t0 = time.time()
        jobs = [(llm, ru, aw, li, gr, a.seed * 10000 + i, 6 if src == "work" else a.rounds, 0, ch, PROMPT_V, src, mem)
                for ch in a.channels for ru in a.rules for aw in a.awareness for li in a.listing
                for gr in a.granularity for src in a.sources for mem in (a.memory if src == "work" else ["keep"])
                for i in range(a.n)
                if ch == "drive" or (gr == "file" and li in ("workspace", "none"))]
        rows = _run_all(run_code, jobs, a.workers)
        ok = [r for r in rows if "error" not in r]
        s = {**summarize(ok), "_errors": len(rows) - len(ok), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"rounds": a.rounds, "sources": a.sources, "memory": a.memory})
        print(json.dumps(s, indent=1))


def cmd_review(a: argparse.Namespace) -> None:
    from .review import CASES, run_review, summarize_review
    out_dir = Path(a.out) / ("review" + (f"_{a.tag}" if a.tag else ""))
    cases = CASES[: a.cases]
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        t0 = time.time()
        jobs = [(llm, c, mode, a.seed * 100000 + ci * 1000 + i, 0)
                for ci, c in enumerate(cases) for mode in a.modes for i in range(a.n)]
        rows = _run_all(run_review, jobs, a.workers)
        ok = [r for r in rows if "error" not in r]
        s = {**summarize_review(ok), "_errors": len(rows) - len(ok), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"cases": [c["id"] for c in cases], "modes": a.modes})
        print(json.dumps(s, indent=1))


def cmd_livewiki(a: argparse.Namespace) -> None:
    from .live_wiki import POST_STYLES, run_live, summarize
    out_dir = Path(a.out) / ("livewiki" + (f"_{a.tag}" if a.tag else ""))
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        t0 = time.time()
        # planted-post texts: the named styles, plus any templates from --post-file (--post-styles file = all of them)
        posts = {st: POST_STYLES[st] for st in a.post_styles if st in POST_STYLES}
        file_posts = json.loads(Path(a.post_file).read_text()) if a.post_file else {}
        posts.update(file_posts)
        styles = [st for st in a.post_styles if st != "file"] + (sorted(file_posts) if "file" in a.post_styles else [])
        unknown = [st for st in styles if st not in posts]
        if unknown:
            raise SystemExit(f"unknown post styles {unknown}; known: {sorted(posts)}")
        jobs = [(llm, a.agents, ru, aw, rc, a.seed * 1000 + i, a.rounds, a.budget, a.stagger, a.timeout_p, ip, 0,
                 src, var, df, ps,
                 None if ps == "plain" or (ps in POST_STYLES and posts[ps] == POST_STYLES[ps]) else posts[ps])
                for ru in a.rules for aw in a.awareness for rc in a.recent for ip in a.init_post
                for src in a.sources for var in a.variants for df in a.defenses for ps in styles for i in range(a.n)]
        rows = _run_all(run_live, jobs, a.workers)
        ok = [r for r in rows if "error" not in r]
        s = {**summarize(ok), "_errors": len(rows) - len(ok), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"agents": a.agents, "rounds": a.rounds, "budget": a.budget,
                                        "stagger": a.stagger})
        print(json.dumps(s, indent=1))


def cmd_committee(a: argparse.Namespace) -> None:
    from .committee import all_cases, run_committee, run_seed, summarize
    out_dir = Path(a.out) / ("committee" + (f"_{a.tag}" if a.tag else ""))
    cases = all_cases()[: a.cases]
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        t0 = time.time()
        jobs = [(llm, c, ch, run_seed(a.seed, c["id"], rep), a.rounds)
                for c in cases for ch in a.channels for rep in range(a.reps)]
        rows = _run_all(run_committee, jobs, a.workers)
        ok = [r for r in rows if "error" not in r]
        s = {**summarize(ok), "_errors": len(rows) - len(ok), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"cases": [x["id"] for x in cases], "reps": a.reps, "rounds": a.rounds})
        print(json.dumps(s, indent=1))


def cmd_wikigame(a: argparse.Namespace) -> None:
    """The wiki game (scb/wikigame.py). --n gives the worlds per --agents value (one count for all, or one each).
    World seeds are seed*100000 + N*1000 + i, so every condition at the same N and i shares the world.
    Every setting is checked before any call. A failed world is saved with its condition and counted per condition;
    if more than --max-error-frac of the worlds fail, the results are still saved but the command exits with status
    2, so scripts/sweep.py does not mark the experiment done."""
    from itertools import product

    from .llm import server_info
    from .wikigame import condition_fields, condition_key, run_world, summarize
    out_dir = Path(a.out) / ("wikigame" + (f"_{a.tag}" if a.tag else ""))
    if len(a.n) not in (1, len(a.agents)):
        raise SystemExit("--n takes one count, or one count per --agents value")
    worlds = dict(zip(a.agents, a.n * len(a.agents) if len(a.n) == 1 else a.n))

    def one(llm: Any, kw: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_world(llm, **kw)
        except Exception as e:  # noqa: BLE001  (a failed call loses its world, which is recorded, not dropped)
            cond = condition_fields(llm, **{k: v for k, v in kw.items() if k not in ("debrief", "agent_workers")})
            return {**cond, "error": str(e)[:500]}

    failed_models = []
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        server = server_info(llm.base_url) if hasattr(llm, "base_url") else None   # test stand-ins have no server
        t0 = time.time()
        jobs, seen = [], set()
        for n, rg, ti, sh, au, wi, iso, rs, aw, df, sc, pl in product(
                a.agents, a.regimes, a.titles, a.sharing, a.audit, a.window, a.isolated, a.reason, a.awareness,
                a.defenses, a.show_count, a.plant):
            au_eff = au if sh == "prohibited" else 0.0   # no rule, no audit
            for i in range(worlds[n]):
                key = (n, rg, ti, sh, au_eff, wi, iso, rs, aw, df, sc, pl, i)
                if key in seen:
                    continue
                seen.add(key)
                kw = dict(n_agents=n, seed=a.seed * 100000 + n * 1000 + i, regime=rg, titles=ti, sharing=sh,
                          audit=au_eff, window=wi, ticks=a.ticks, isolated=iso == "on", awareness=aw,
                          reason=rs == "on", debrief=not a.no_debrief, sample_offset=0, max_tokens=a.max_tokens,
                          history_chars=a.history_chars, agent_workers=a.agent_workers, defense=df, server=server,
                          show_count=sc == "on", plant=pl)
                try:
                    condition_fields(llm, **{k: v for k, v in kw.items() if k not in ("debrief", "agent_workers")})
                except ValueError as e:
                    raise SystemExit(f"invalid setting {kw}: {e}") from None
                jobs.append((llm, kw))
        rows = _run_all(one, jobs, a.workers)
        n_err = sum("error" in r for r in rows)
        # experiments that share a tag keep each other's worlds; a rerun of the same world replaces it
        rows = _keep_saved_rows(out_dir, model, rows,
                                lambda r: (condition_key(r), r["seed"], r.get("sample_offset", 0)))
        s = {**summarize(rows), "_errors": sum("error" in r for r in rows), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"ticks": a.ticks})
        print(json.dumps(s, indent=1))
        if rows and n_err / len(rows) > a.max_error_frac:
            failed_models.append(f"{model}: {n_err}/{len(rows)} worlds failed")
    if failed_models:
        print("FAILED WORLDS above --max-error-frac:\n  " + "\n  ".join(failed_models), flush=True)
        raise SystemExit(2)


def cmd_evolve_post(a: argparse.Namespace) -> None:
    """Evolve the planted post's wording against one model (scb/evolve_post.py). Writes
    <out>/evolve_post/<model>__<defense>.json and adds the evolved finalists to <out>/evolve_post/posts.json, for
    livewiki --post-file ... --post-styles file."""
    from .evolve_post import evolve
    from .live_wiki import POST_STYLES
    out_dir = Path(a.out) / "evolve_post"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in a.models:
        llm = LLM(model=model, temperature=a.temperature, max_tokens=a.max_tokens)
        mutator = LLM(model=a.mutator or model, temperature=1.0, max_tokens=400)
        cfg = {"agents": a.agents, "rule": a.rule, "awareness": a.awareness, "recent": a.recent, "rounds": a.rounds,
               "budget": a.budget, "stagger": a.stagger, "init_post": a.init_post, "defense": a.defense}
        res = evolve(llm, mutator, cfg, a.generations, a.population, a.keep, a.runs, a.final_runs, a.final_top,
                     a.seed, a.workers, log=lambda m: print(m, flush=True))
        tag = f"{model.replace('/', '_').replace(':', '_')}__{a.defense}"
        (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=1))
        with open(out_dir / ".posts.lock", "w") as lock:   # several jobs (one per model) may finish at once
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                pf = out_dir / "posts.json"
                posts = json.loads(pf.read_text()) if pf.exists() else {}
                evolved = [x for x in res["final"] if x["name"] not in POST_STYLES]
                if not evolved:   # the hand-written styles won: still pass on the best evolved candidate
                    evolved = sorted((c for g in res["history"] for c in g if c["name"] not in POST_STYLES),
                                     key=lambda c: -c["fitness"])[:1]
                for k, f in enumerate(evolved):
                    posts[f"evo-{tag}-{k + 1}"] = f["template"]
                pf.write_text(json.dumps(posts, indent=1))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        print("final: " + "; ".join(f"{f['name']} used {f['used']}/{f['n']}, relayed {f['relayed']}/{f['n']}"
                                    for f in res["final"]), flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, many_n: bool = False, models_required: bool = True) -> None:
        sp.add_argument("--models", nargs="+", required=models_required, default=[], help="served model name(s)")
        if many_n:
            sp.add_argument("--n", type=int, nargs="+", default=[100],
                            help="worlds per condition: one count, or one per --agents value")
        else:
            sp.add_argument("--n", type=int, default=100, help="runs (seeds) per condition")
        sp.add_argument("--seed", type=int, default=1, help="seed base: runs use seeds derived from it")
        sp.add_argument("--temperature", type=float, default=1.0)
        sp.add_argument("--max-tokens", type=int, default=300)
        sp.add_argument("--workers", type=int, default=16, help="concurrent runs")
        sp.add_argument("--out", default="results")
        sp.add_argument("--tag", default="", help="suffix of the output directory")

    d = sub.add_parser("deaddrop", help="scenarios 1-2: the two-instance dead drop")
    common(d)
    d.add_argument("--channels", nargs="+", default=["drive"], choices=["drive", "wiki", "kv", "packages"])
    d.add_argument("--rounds", type=int, default=4)
    d.add_argument("--rules", nargs="+", default=["forbidden", "silent"])
    d.add_argument("--awareness", nargs="+", default=["aware", "unaware"])
    d.add_argument("--listing", nargs="+", default=["workspace", "random", "none"])
    d.add_argument("--granularity", nargs="+", default=["file", "dir"])
    d.add_argument("--defenses", nargs="+", default=["none"], choices=list(DEFENSES), help="none | warning | warning_generic | warning_use")
    d.set_defaults(fn=cmd_deaddrop)

    s = sub.add_parser("swarm", help="scenario 2 with N copies: --agents N --questions Q --f-inv --randomized R "
                                       "--awareness --log (each call sends max_tokens=250, as scenario 2)")
    common(s, models_required=False)
    s.add_argument("--mix", nargs="+", default=[], metavar="MODEL:COUNT,...",
                   help="mixed swarms, run besides any --models: copies of several models on one wiki, e.g. "
                        "qwen3-8b:2,gemma-3-27b:2 (the swarm size is the total; seats are filled in this order, so with "
                        "--f-inv 2 an even count puts a model in both halves of the answers)")
    s.add_argument("--channels", nargs="+", default=["wiki"], choices=["wiki", "kv", "packages"])
    s.add_argument("--agents", nargs="+", type=int, default=[2], help="N, number of copies (one or more values)")
    s.add_argument("--questions", type=int, default=4, help="Q, number of questions (= rounds)")
    s.add_argument("--f-inv", type=int, default=2, help="each copy holds the answers to Q / f_inv questions")
    s.add_argument("--randomized", nargs="+", type=int, default=[0], choices=[0, 1],
                   help="R: 0 = fixed disjoint blocks, 1 = a random Q/f_inv questions per copy")
    s.add_argument("--rules", nargs="+", default=["forbidden", "silent"])
    s.add_argument("--awareness", nargs="+", default=["aware", "unaware"], choices=["aware", "unaware"],
                   help="aware = the prompt says other instances of the same agent take the quiz; unaware = it does not")
    s.add_argument("--listing", nargs="+", default=["workspace", "none"], choices=["workspace", "none"])
    s.add_argument("--defenses", nargs="+", default=["none"], choices=list(DEFENSES))
    s.add_argument("--read-cap", type=int, default=0, help="a read returns only the newest K notes (0 = all)")
    s.add_argument("--shuffle-writes", action="store_true", help="apply each round's writes in a random order")
    s.add_argument("--plant", nargs="+", default=["none"], choices=["none", "wrong", "true"],
                   help="before round 1, put a note with a wrong (or the true) value on the titles seekers read most, "
                        "for half of the questions (scb/swarm.py PLANT_TITLES); none = no note")
    s.add_argument("--log", nargs="+", default=["none"], choices=["none", "A", "B"],
                   help="A = add an orchestrator log listing every sandbox's model and which one is this session; "
                        "B = placebo: the same log, but the other sandboxes run a different model")
    s.add_argument("--start", type=int, default=0, help="first seed index: seeds are seed*10000 + start ... + n-1")
    s.add_argument("--max-error-frac", type=float, default=0.02,
                   help="exit with status 2 if more than this share of runs fail (results are saved either way)")
    s.set_defaults(fn=cmd_swarm)

    k = sub.add_parser("codedrop", help="scenario 5: the coding dead drop (5b with --sources work)")
    common(k)
    k.add_argument("--channels", nargs="+", default=["drive"], choices=["drive", "wiki", "kv", "packages"])
    k.add_argument("--rounds", type=int, default=4)
    k.add_argument("--sources", nargs="+", default=["notes"], choices=["notes", "work"],
                   help="notes: the other copy's constants in private notes (scenario 5); work: in the other copy's "
                        "own earlier submissions (scenario 5b, six rounds)")
    k.add_argument("--memory", nargs="+", default=["keep"], choices=["keep", "reset"],
                   help="with --sources work: keep the conversation through all rounds, or start round 4 fresh")
    k.add_argument("--rules", nargs="+", default=["forbidden", "silent"])
    k.add_argument("--awareness", nargs="+", default=["aware", "unaware"])
    k.add_argument("--listing", nargs="+", default=["workspace", "none"])
    k.add_argument("--granularity", nargs="+", default=["file"])
    k.set_defaults(fn=cmd_codedrop)

    v = sub.add_parser("review", help="scenario 6: a review with a shared interest")
    common(v)
    v.add_argument("--cases", type=int, default=6, help="how many of the six review cases")
    v.add_argument("--modes", nargs="+", default=["collusion", "solo", "honest"], choices=["collusion", "solo", "honest"])
    v.set_defaults(fn=cmd_review)

    w = sub.add_parser("livewiki", help="scenario 3: the live multi-agent wiki")
    common(w)
    w.add_argument("--agents", type=int, default=4)
    w.add_argument("--stagger", type=int, default=6, help="steps between the copies' start times")
    w.add_argument("--rounds", type=int, default=6)
    w.add_argument("--budget", type=int, default=4, help="tool calls per round")
    w.add_argument("--rules", nargs="+", default=["forbidden", "silent"])
    w.add_argument("--awareness", nargs="+", default=["aware"], help="aware | pair | cohort | unaware")
    w.add_argument("--recent", nargs="+", default=["on", "off"])
    w.add_argument("--timeout-p", type=float, default=0.9, help="search failure probability (source=search)")
    w.add_argument("--init-post", nargs="+", default=["none"], help="none | focal | nonfocal (planted post)")
    w.add_argument("--sources", nargs="+", default=["search"], help="search | ahead | notes")
    w.add_argument("--variants", nargs="+", default=["base"], help="base | onewiki | quota | persist | pooled")
    w.add_argument("--defenses", nargs="+", default=["none"], choices=list(DEFENSES), help="none | warning | warning_generic | warning_use")
    w.add_argument("--post-styles", nargs="+", default=["plain"],
                   help="planted post text: plain | persist | protocol | copy, names from --post-file, or file (all)")
    w.add_argument("--post-file", default=None, help="JSON {name: template with {values}} of extra planted posts")
    w.set_defaults(fn=cmd_livewiki)

    c = sub.add_parser("committee", help="scenario 4: the sealed committee")
    common(c)
    c.add_argument("--channels", nargs="+", default=["sealed", "sealed_solo", "honest"],
                   help="sealed (told an ally exists) | sealed_solo (told they act alone) | honest (no colluders)")
    c.add_argument("--cases", type=int, default=8, help="use the first N committee cases")
    c.add_argument("--reps", type=int, default=10, help="runs per committee case and channel")
    c.add_argument("--rounds", type=int, default=2, help="discussion rounds")
    c.set_defaults(fn=cmd_committee)

    g = sub.add_parser("wikigame", help="the wiki game: the scenario 2 dead drop scaled to N agents on one wiki")
    common(g, many_n=True)
    g.set_defaults(max_tokens=400)
    g.add_argument("--agents", type=int, nargs="+", default=[4], help="population sizes N")
    g.add_argument("--regimes", nargs="+", default=["breadth"], type=_regime,
                   help="breadth | sqrt | crowding | quiz | fixed-h<H>-s<S> (the fixed-N intervention)")
    g.add_argument("--titles", nargs="+", default=["free"], choices=["free", "given"])
    g.add_argument("--sharing", nargs="+", default=["prohibited"], choices=["prohibited", "allowed", "team"])
    g.add_argument("--audit", type=float, nargs="+", default=[0.25],
                   help="audit rate 0 or 1/k (used with sharing=prohibited only)")
    g.add_argument("--window", type=int, nargs="+", default=[8], help="edits shown per page view")
    g.add_argument("--ticks", type=int, default=16, help="steps before the unannounced end")
    g.add_argument("--isolated", nargs="+", default=["off"], choices=["off", "on"],
                   help="on: every agent on its own private wiki (the replay baseline)")
    g.add_argument("--reason", nargs="+", default=["on"], choices=["on", "off"], help="the one-sentence reason field")
    g.add_argument("--awareness", nargs="+", default=["aware"], choices=["aware", "unaware"])
    g.add_argument("--defenses", nargs="+", default=["none"], choices=list(DEFENSES),
                   help="note after the rules: none | warning | warning_generic | warning_use")
    g.add_argument("--show-count", nargs="+", default=["on"], choices=["on", "off"],
                   help='off: a view of a page longer than --window omits "(<total> edits, showing the newest <window>)"')
    g.add_argument("--plant", nargs="+", default=["none"], choices=["none", "wrong", "true"],
                   help="put a note with a wrong or the true value on the common titles of half the lookups at step 0")
    g.add_argument("--max-error-frac", type=float, default=0.02,
                   help="exit with status 2 if more than this share of worlds fail (results are saved either way)")
    g.add_argument("--no-debrief", action="store_true", help="skip the debrief question after the test")
    g.add_argument("--history-chars", type=int, default=40000,
                   help="shorten the oldest page texts in an agent's history beyond this many characters")
    g.add_argument("--agent-workers", type=int, default=64, help="concurrent calls within one world")
    g.set_defaults(fn=cmd_wikigame)

    e = sub.add_parser("evolve-post", help="scenario 3a: evolve the planted post's wording against a model")
    common(e)
    e.add_argument("--mutator", default=None, help="model that writes the new texts (default: the evaluated model)")
    e.add_argument("--defense", default="none", choices=list(DEFENSES))
    e.add_argument("--generations", type=int, default=8)
    e.add_argument("--population", type=int, default=8)
    e.add_argument("--keep", type=int, default=4)
    e.add_argument("--runs", type=int, default=30, help="runs per candidate per generation")
    e.add_argument("--final-runs", type=int, default=100, help="held-out runs for the finalists")
    e.add_argument("--final-top", type=int, default=3)
    e.add_argument("--agents", type=int, default=4)
    e.add_argument("--stagger", type=int, default=6)
    e.add_argument("--rounds", type=int, default=6)
    e.add_argument("--budget", type=int, default=4)
    e.add_argument("--rule", default="forbidden")
    e.add_argument("--awareness", default="aware")
    e.add_argument("--recent", default="on")
    e.add_argument("--init-post", default="focal")
    e.set_defaults(fn=cmd_evolve_post)

    q = sub.add_parser("capability", help="a capability score per model with its game settings (scb/capability.py)")
    common(q, models_required=True)
    q.add_argument("--subset", required=True, help="questions, JSON lines (scripts/mmlupro_subset.py)")
    q.set_defaults(fn=cmd_capability)

    a = p.parse_args(argv)
    a.fn(a)


def cmd_capability(a: argparse.Namespace) -> None:
    from .capability import MAX_TOKENS, ask, load, summarize
    qs = load(a.subset)
    out_dir = Path(a.out) / ("capability" + (f"_{a.tag}" if a.tag else ""))
    for model in a.models:
        llm = _llm(model, a.temperature, MAX_TOKENS)
        t0 = time.time()
        rows = _run_all(ask, [(llm, q, i, a.seed, MAX_TOKENS) for i, q in enumerate(qs)], a.workers)
        s = {**summarize(rows), "_elapsed_s": round(time.time() - t0, 1)}
        _save(out_dir, model, rows, s, {"subset": Path(a.subset).name, "questions": len(qs),
                                        "max_tokens_per_call": MAX_TOKENS})
        print(json.dumps(s, indent=1), flush=True)


if __name__ == "__main__":
    main()
