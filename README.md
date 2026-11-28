# Code and results for "More Copies, More Collusion: Understanding Schelling Point Collusion in LLM Swarms"

This repository holds everything behind the numbers in the paper: the game's code, every run the paper reports, and
the scripts that turn the runs into the paper's figures and tables. The paper's figures are drawn in LaTeX from numbers
typed into its source; each of those numbers comes from one of the write-ups in `analysis/`.

## Layout

| Folder | What it holds |
|---|---|
| `scb/`, `configs/`, `scripts/`, `tests/` | The benchmark: the 4-round swarm game (`scb/swarm.py`, `scb/wikigame.py`), the model clients and the run definitions (`configs/experiments.yaml`). See `BENCHMARK.md` |
| `scaling/` | The scaling study: its scripts (`scaling/scripts/`), the runner patches they need (`scaling/patches/`), and its runs. `scaling/README.md` describes them |
| `scaling/results/` | Stage-1 runs (Qwen3-8B and Gemma 3 27B at N = 2-32, the size comparison, the mixes of two models) and the later tests (planted notes and pages, the rule's halves, DeepSeek V4.1 Flash with 64 copies, the twin wording) |
| `scaling/aaai/`, `scaling/aaai_or/`, `scaling/aaai_claude/` | Runs made for this paper: N = 64 and 128, Gemma alone at N = 16-128, the N = 16 mixes, the rule ablation, the capability quiz, and the models reached through OpenRouter and the Anthropic API |
| `scaling/aaai_pilots/` | Two-game pilots of the OpenRouter and Claude runs, made to check settings before the full runs. Nothing in the paper uses them, and `run_all.sh` does not read them |
| `wiki_game/data/swarm4round/` | Qwen3-4B, 14B and 32B at N = 2-32 and Qwen3-8B reruns, all with 2 answers per copy (pod A). Kept at the path the hackathon repository uses, which several of its scripts expect |
| `runs/` | Five run folders kept apart from `scaling/`, and the scripts that launched the runs (`runs/launch/`); `runs/README.md` says why |
| `analysis/` | The scripts that write the paper's numbers, their write-ups (`*.md`) and machine-readable output (`out/`) |

Each run folder holds `summary.json` and `raw/<model>.jsonl.gz`: one record per game, with every model call, the pages
each copy wrote and read, and the answers it gave.

## Rebuilding the numbers

With Python 3.12 and numpy, scipy, pandas and matplotlib:

```bash
bash analysis/run_all.sh
```

It reruns every script in order and compares each write-up with the version the paper was written from
(`analysis/expected/`). It then reruns the hackathon repository's own scripts and compares each JSON file they write
with the committed one in `scaling/results/`. On 28 November 2026, after the rescoring below, every write-up was reproduced byte for byte. There were three
expected exceptions:

- `predictions_mix3.md`, recomputed, names a different commit of the code it imports, because it reads that hash from
  git. Its numbers are the same.
- `law_points.py` leaves out DeepSeek V4.1 Flash with 32 and 64 copies, which `law_wrong_aaai.json` now contains:
  the paper scores those runs against their own registered forecast (app:deepseek), not in fig:law.
- `byround.md` and `scaling_tests.md` are written by hand. `run_all.sh` checks `byround.md` against `byround.json`;
  `scaling_tests.md` names the result file behind each number of the green text.

## Rescoring with the decimal answer rule (28 November 2026)

An answer is correct when it is within 0.1 of the true value. The game first tested this in binary floating point,
`abs(answer - truth) <= 0.1`, which rejects some answers exactly 0.1 away (6.3 for a true 6.4). `scb/quiz.py`
`within_tenth()` now compares in decimals, and `scripts/rescore_runs.py` rescored every saved game in place: 2,401
answers in all, with `answer`, `truth` and every model call unchanged. The copies never see whether an answer was
scored correct, so only the scoring needed redoing. `analysis/rescore.md` lists the answers rescored and the collusion
rate before and after for every condition; the largest change is Qwen3-14B with 128 copies, 41.55% to 42.16%. Every
file in `scaling/results/*.json` and every write-up in `analysis/` was then rebuilt from the rescored games.

The same rebuild brings in the prediction's use rate for a model with no finds in pairs (hackathon repository commit
e96acca: the use rate is unmeasured, and set to 1, not 0; only Qwen3-32B with the rule moves), and the tests of notes
rewritten by other models (`scaling/` items 28-30, `results/{paraphrase,writers,grid,rewrites_extra}.json`).

## Where each part of the paper comes from

| Paper part | Write-up in `analysis/` | Script |
|---|---|---|
| Figures and tables kept from the hackathon version (fig:salience, fig:r1 to N = 32, fig:r2, fig:r3, tab:titlematch, tab:habits, tab:r3seekers) | `scaling/README.md` | `scaling/scripts/figures_*.py`, `predict_swarm.py`, `analyze_mixed.py` |
| fig:r1, fig:models, fig:r2, tab:sizes, tab:r2steps at N = 64 and 128; Gemma 3 27B at N = 16-128 | `larger_n.md` | `larger_n.py` |
| Guesses (sec:law), fig:guess, tab:roles, tab:counts; the rule, fig:rule, tab:rule | `roles.md` | `roles.py` |
| sec:rulemodels, fig:rulemodels, tab:rulemodels (CR) | `rule_models.md` | `rule_models.py`, `rule_models_tex.py` |
| tab:rule and tab:rulemodels: the paired intervals of the difference | `rule_models_paired.md` | `rule_models_paired.py` |
| 8B's seekers that opt out (app:optout, tab:optout) | `seeker_optout.md` | the working-notes scripts it names; not rerun here |
| Green text, notes rewritten by other models (app:writers, tab:writers) | `scaling/README.md` items 28-30 | `scaling/scripts/analyze_{paraphrase,writers,grid,rewrites_extra}.py` |
| Rescoring: answers rescored and CR before and after | `rescore.md` | `scripts/rescore_runs.py` |
| fig:law, tab:law, tab:lawsizes; the medians in sec:law, sec:misses, sec:r3 | `law_points.md` | `law_points.py` on `scaling/results/{law_wrong,title_law}_aaai.json` (from `scaling/scripts/law_wrong.py`, `title_law.py`) |
| fig:missrounds; use rates by round half | `byround.md` | `byround.py` |
| fig:rounds; late holders after a page with notes | `late_holders.md` | `late_holders.py` |
| fig:lateholders | `rounds.md` | `rounds.py` |
| N = 16 mixes: fig:r3, sec:r3, tab:r3full | `predictions_mix16.md`, `results_mix16.md` | `predict_mix16.py`, `score_mix16.py` |
| tab:gram; title coverage | `mixtures.md`, `coverage.md` | `mixtures.py`, `coverage.py` |
| fig:simplex, app:mix | `predictions_mix3.md`, `results_mix3.md`, `fig14_points.md` | `predict_mix3.py`, `score_mix3.py`, `fig14_points.py` |
| The planted note (sec:misses); registered forecasts at N = 64 and 128 | `predictions_pod5.md`, `results_pod5.md` | `fuller.py`, `predict_pod5.py`, `score_pod5.py` |
| DeepSeek seekers' use by the most common value on the page (sec:misses) | `agreement.md` | `agreement.py` |
| Green text: fig:precedent, tab:precedent, tab:pagecontent, tab:deepseek, tab:twin, tab:twinpage, tab:rulehalves, tab:registered | `scaling_tests.md` | `scaling/scripts/analyze_{ablations,plant,transplant,twin,page_content,twin_page}.py` |
| Gemma's posting by notes read, the filler runs (sec:misses) | `scaling/README.md` items 16-17 | `scaling/scripts/analyze_posting.py`, `holders_seekers.py` |

## Registered forecasts

`predictions_mix16.md`, `predictions_mix3.md` and `predictions_pod5.md` were written and committed before the runs they
forecast. The `predict_*.py` scripts refuse to overwrite them; `run_all.sh` recomputes them and checks that they match.
The forecasts in `scaling/results/predictions_*.md` were registered the same way in the hackathon repository.

## Provenance

- Benchmark: LidaSafety/schelling-collusion-benchmark, commit 2e0160d (branch `mixed-3way`), which contains the branches
  `wiki-game`, `mixed-n16` and `claude-api`.
- `scaling/`: LidaSafety/Swarm_schelling_hackathon, `scaling/` at branch `scaling` f3da57c (items 1-30), with
  `scripts/law_wrong.py` and `title_law.py` from `main` e96acca, and the result folders of branch `aaai` (ba7ee33) in
  `scaling/aaai*`. Its result files were then rebuilt from the rescored games (see above).
- `wiki_game/data/swarm4round`: the same repository, the same path.
