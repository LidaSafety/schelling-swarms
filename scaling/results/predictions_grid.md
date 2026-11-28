# Forecasts for the reader-by-writer grid (does each model respond most to its own model's text?)

Written before these runs start, and committed before the first run.

## Why

In items 28 and 29, answer pages were rewritten note by note by different models, keeping the state, the value and
about the length. Gemma 3 27B's late holders posted most after Gemma's rewrite (51.6%), less after Qwen3-8B's
(35.8%) and barely more than with nothing planted after Llama 3.3 70B's (26.4% against 24.1%). That is the order of
how often Gemma says it wrote each text. Qwen3-8B's late holders responded most to Gemma's rewrite and alike to their
own and Llama's. Two explanations fit:
- *self-preference*: each model responds most to its own model's text (seen in Gemma, not in Qwen3-8B);
- *writer effect*: Gemma's text moves any reader more.

With Gemma as both the reader that responds to its own text and the writer whose text moves Qwen3-8B, the two are
confounded. A third model that is both a reader and a writer gives a 3 × 3 grid that separates them. Each writer
appears once on the diagonal, so in the diagonal index below a writer's general influence and a reader's general
responsiveness cancel out, and only a match between reader and writer remains.

The planned third reader, Llama 3.3 70B, cannot serve: in 10 pairs on the fixed prompt (`plant_reader_llama_smoke10`)
its late holders read a page in round 1 in 1 game, and its holders posted in 38 of 40 rounds. A page planted in round
1 would almost never be read. gpt-oss-120b can: in its 100 pairs with the rule (`or_pairs`), 97% of its late holders
read a page in round 1, 68% of those under the state's name alone, and 49% post their round-2 answer.

## Runs

| Reader | How it plays | Page it reads | Games |
|---|---|---|---|
| Gemma 3 27B | pod, as in items 28–29 | its own game's answer page | 4 arms × 300 (sample offsets 6–8) |
| Qwen3-8B (thinking off) | pod, as in items 28–29 | its own game's answer page | 4 arms × 400 (offsets 8–11) |
| gpt-oss-120b | OpenRouter, as in its `or_pairs` runs: one host (Crusoe), no fallbacks, reasoning effort low, 6× the 250-token budget | Gemma's game's answer page, moved to its usual round-1 title, the state's name | 4 arms × 400 (offsets 0–3) |

Every reader plays pairs on the fixed prompt (unaware, rule in force, no page list, `--log none`, 4 questions, 2
answers per copy, fixed blocks, temperature 1), seeds 10000–10099, with sample offsets new to these quizzes. That is
4,400 games. The arms are nothing, or the page rewritten note by note by each writer:
- Gemma 3 27B and Qwen3-8B: item 28's pages (`results/paraphrase/<page>_by_gemma.json`, `_by_q8.json`);
- gpt-oss-120b: new pages (`scripts/make_paraphrase_pages.py --writers gptoss`), with item 28's prompt and checks and
  the reply's first line that carries a number, made with the model served as for its runs as a player and
  committed before the first pair run.

The gpt-oss-120b reader's pages are Gemma's page rewritten by each writer, with the title replaced by the state's
name and the notes unchanged (`scripts/make_reader_pages.py`). Runs: `scripts/run_grid_pairs.sh gemma-3-27b` and
`... qwen3-8b` (tags `plant_grid_gemma`, `plant_grid_q8`), and `scripts/run_reader_pairs.sh` (`plant_grid_gptoss`).

The outcome is items 26–29's: among the late holders whose round-1 read was the planted title (with nothing planted,
the same title), the share that post their round-2 answer in round 2. For reader r and writer w, the effect `e[r][w]`
is the rate after w's page minus the rate with nothing planted. Answers raise some readers' posting and lower others',
so each reader's effects are read in its own direction: `s[r]` is the sign of the mean of its three effects. Then:
- **own-text advantage** `SP[r] = s[r] · (e[r][own] − mean of e[r][w] over the other two writers)`;
- **diagonal index** `D` = the mean of `SP` over the three readers;
- **writer effect** `W` = for Qwen3-8B and gpt-oss-120b, the readers for whom Gemma is not their own writer, the mean
  of `s[r] · (e[r][Gemma] − e[r][the third writer])`.

Intervals use 2,000 resamples of the seeds, with all games of a seed, for all readers and arms, moving together.
Analysis: `scripts/analyze_grid.py`, written before the runs and checked end to end on scripted stand-ins.

Manipulation check, reported, not registered: each reader is asked, of 150 notes from each page it reads, whether it
or a different AI model wrote each (`scripts/check_authorship.py --by gemma q8 gptoss --with-gptoss`).

## Forecasts

From items 28–29, % posting with nothing / Gemma's / Qwen3-8B's page: Gemma 18.5–24.1 / 49.8–51.6 / 35.8; Qwen3-8B
81.5–82.0 / 42.3–52.6 / 55.1–59.1. gpt-oss-120b posts 49% in its pairs.

| Explanation | Predicts |
|---|---|
| Self-preference | `D` above zero; `SP` above zero for gpt-oss-120b too |
| Writer effect only | `W` above zero, `D` near zero |
| Both | `D` and `W` above zero |

Our guesses: `SP` about +15 points for Gemma, −3 for Qwen3-8B, +5 for gpt-oss-120b, so `D` about +6; `W` about +5.

**Registered.** Each test is met when its 95% interval lies wholly above zero.
- (a) Self-preference: `D` above zero.
- (b) gpt-oss-120b's own-text advantage `SP[gpt-oss-120b]` above zero.
- (c) Writer effect: `W` above zero.

Readings:
- **(a) met, (c) not:** models respond most to their own model's text, and Gemma's text is not more influential in
  general.
- **(c) met, (a) not:** Gemma's text moves every reader more, and there is no reader-writer match beyond that;
  Gemma's response to its own text in items 28–29 was the writer effect.
- **(a) and (c) met:** both.
- **Neither:** no consistent reader-by-writer pattern at this size.
- **(b):** whether a model other than Gemma responds most to its own text. If a reader's mean effect has an interval
  that includes zero, its direction is not identified, and its `SP` is reported but not read.

Reported, not registered:
- the 3 × 3 table and each reader's `SP`;
- each writer's mean effect across readers;
- the manipulation check;
- the rewritten pages' lengths and bare-number shares.

Not covered: more readers and writers, requests, and rewriting inside a running swarm.
