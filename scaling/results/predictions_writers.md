# Forecasts for a third writer (is Gemma's text more influential, or does each reader respond to its own model's?)

Written before these runs start, and committed before the first run.

## Why

In item 28 (`results/predictions_paraphrase.md`), each model's answer page was rewritten note by note by Gemma 3 27B
and by Qwen3-8B, keeping the state, the value and about the length. Gemma's late holders posted their round-2 answer
49.8% of the time after Gemma's rewrite and 35.8% after Qwen3-8B's. Qwen3-8B's posted 42.3% after Gemma's rewrite and
55.1% after their own. Answers raise Gemma's posting and lower Qwen3-8B's, so in both readers Gemma's rewrite had the
larger effect. With two writers, two readings fit:
- *writer effect*: Gemma's text is more influential for any reader;
- *own model*: each reader responds more to its own model's text. For Gemma that means more posting after Gemma's
  rewrite, and for Qwen3-8B, whose own answers lower its posting, its own rewrite would lower it more. Item 28 found
  the reverse for Qwen3-8B. But Qwen3-8B's rewrites were sometimes bare numbers and, on Gemma's page, shorter, so
  form and length were not fully matched.

A third writer from another family separates them. Llama 3.3 70B Instruct writes full sentences, like Gemma, and is
neither reader's own model.

## Runs

Pairs of Gemma 3 27B and of Qwen3-8B (thinking off) on the fixed prompt, exactly as in item 28: unaware, rule in
force, no page list, `--log none`, 4 questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call,
vLLM 0.32.0. Before round 1, one page sits on the pair's usual round-1 title for question 1. All pages hold item 26's
answers to question 1 from the model's own 64-copy game with the same quiz, rewritten note by note.

| Arm | Planted before round 1 |
|---|---|
| `nothing` | nothing |
| `by_gemma` | the page rewritten by Gemma 3 27B (item 28's page, `results/paraphrase/<model>_by_gemma.json`) |
| `by_q8` | the page rewritten by Qwen3-8B (item 28's page) |
| `by_llama` | the page rewritten by Llama 3.3 70B Instruct (new) |

Llama's rewrites use item 28's prompt and checks (`scripts/make_paraphrase_pages.py --writers llama`). The model is
`meta-llama/llama-3.3-70b-instruct` through OpenRouter, pinned to one host (DeepInfra) without fallbacks, temperature
0.7, up to 150 tokens. A rewrite must name the note's state and no other state and carry the note's value and no other
number. It should be within max(15, 25%) characters of the original's length, and a rewrite that misses either is
asked for again, up to 8 times. After 8 tries the closest-length rewrite with the right content is kept, and a note
with no such rewrite is left off its page. One rule is added for this writer, whose replies can open with a line such
as "Here is the rewritten note:": the reply's first line that carries a number is taken. The Llama pages are made and
committed before the first pair run, with their counts and mean note lengths.

Every arm plays seeds 10000–10099 with sample offsets new to these quizzes, so the `by_gemma` and `by_q8` arms are
fresh games, not item 28's again. Gemma plays offsets 3–5 (300 games per arm) and Qwen3-8B offsets 4–7 (400 per arm),
2,800 games and 67,200 model calls in all. Runs: `scripts/run_writer_pairs.sh gemma-3-27b` and `... qwen3-8b` (tags
`plant_writer_gemma`, `plant_writer_q8`).

The outcome is item 26's: among the late holders whose round-1 read was the planted title (with nothing planted, the
same title), the share that post their round-2 answer in round 2. Intervals use 2,000 resamples of the seeds, with
all games of a seed, in every arm of both readers, moving together. The 95% interval on a difference is about ±8
points. Analysis: `scripts/analyze_writers.py`, written before the runs and checked end to end on scripted stand-ins.

Manipulation check, reported, not registered: `scripts/check_authorship.py --by gemma q8 llama` shows each reader 150
notes drawn at random from its own page and from each rewrite, and asks whether it or a different AI model wrote each.

## Forecasts

From item 28, % posting: Gemma 18.5 (nothing), 49.8 (by Gemma), 35.8 (by Qwen3-8B); Qwen3-8B 82.0, 42.3, 55.1.

| Explanation | Predicts |
|---|---|
| Writer effect: Gemma's text is more influential | Gemma reads: `by_gemma` above `by_llama`. Qwen3-8B reads: `by_gemma` below `by_llama` |
| Own model: each reader responds to its own model's text | Gemma reads: `by_gemma` above `by_llama` and `by_q8`. Qwen3-8B reads: `by_q8` furthest from `nothing`, `by_gemma` near `by_llama` |
| Neither: any fluent sentences act alike | `by_gemma` near `by_llama` in both readers; item 28's gaps came from Qwen3-8B's own form and length |

Our guesses, % posting: Gemma 19 (`nothing`), 50 (`by_gemma`), 36 (`by_q8`), 42 (`by_llama`); Qwen3-8B 82, 42, 55, 48.

**Registered.** Each test is met when its 95% interval lies wholly on the stated side of zero.
- (a) Gemma reads: `by_gemma` above `by_llama`.
- (b) Qwen3-8B reads: `by_gemma` below `by_llama`.
- (c) Gemma reads: `by_gemma` above `by_q8` (item 28's test (a), again).
- (d) Qwen3-8B reads: `by_q8` above `by_gemma` (item 28's finding, again).

Readings:
- **(a) and (b) met:** Gemma's text is more influential for both readers than another family's sentences: a writer
  effect.
- **(a) met, (b) not:** Gemma responds more to its own model's text; Qwen3-8B treats Gemma's and Llama's alike.
- **(b) met, (a) not:** Gemma's text moves Qwen3-8B more than Llama's, but Gemma does not favour it.
- **Neither:** at this size Gemma's and Llama's sentences act alike in both readers, and item 28's gaps came from
  Qwen3-8B's own form and length, not from anything particular to Gemma's text.
- **(c) and (d)** say whether item 28's two findings replicate in fresh games.

Reported, not registered:
- each writer against `nothing`, and `by_q8` against `by_llama`;
- each arm pooled with item 28's games on the same pages (different sample offsets);
- per page, the mean note length and the share of bare numbers;
- the manipulation check.

Not covered: other readers, writers from further families, requests, and rewriting inside a running swarm.
