# Forecasts for the twin text at scale (precedent and self-recognition)

Written before these runs start, and committed before the first run.

## Why

Item 23 of the README, from the Qwen3-8B runs with 2–32 copies (`scripts/analyze_twin.py`): the twin text says the
other copies are "copies of you ... the same model, with the same weights, receiving these same instructions". Against
the aware text ("other instances of the same agent"), it makes late holders post their answer 14.5–21.5 points more
often at every size. Late holders hold questions 2 and 4, so they only searched in round 1. The twin text also weakens
how much less a late holder posts the more notes it saw in round 1. Per doubling of the notes, the aware text's posting
changes by −13.7, −4.4 and −3.9 points with 8, 16 and 32 copies, and the twin text's by −5.2, −1.0 and +0.9 (twin
minus aware +8.5 [1.2, 15.9], +3.5 [0.2, 6.5] and +4.8 [1.6, 7.9]). Overall the two texts stay close from 4 copies up
(−5.1 to +8.4 points): the twin text lowers collusion in rounds 1 and 3 and raises it in rounds 2 and 4.

If telling copies that the others are copies of them changes how they react to what earlier copies wrote, its effect
should be largest where pages are crowded, that is, in big swarms. Gemma 3 27B reacts to crowded pages most strongly.
With the unaware text its late holders post their answer 15.5% of the time in pairs, 24% with 4 and 8 copies, 26%
with 16, 41% with 32 and 65% with 64 (16 copies and up on the `aaai` branch). Its copies also gather on one title, so
from 16 copies almost every late holder reads a page with many notes. Gemma has never run with the twin text in this
game. In Qwen3-8B the late holders' boost does not grow with size (+21.5 points with 4 copies, +18.2 with 32), and
twin-text posting is near its ceiling (88.5–92.8%). With the unaware text and 16 copies, Gemma's late holders post
less without the rule than with it (12.8% against 25.9%), so the rule is not simply what holds them back (see H4).

## Runs

- **Gemma 3 27B**, the fixed prompt but the other-copies sentence: the aware text and the twin text in one tag per
  size (`--awareness aware --twin 0 1`). With the rule in force at 2, 4, 8, 16, 32 and 64 copies (tags
  `twin_n<N>_gemma`), and without it (`--rules silent`) at 16 copies (`twinoff_n16_gemma`).
- **Qwen3-8B**, the same with 64 copies and the rule in force (`twin_n64_q8`), thinking off.
- **Shared settings:** 100 games per text and size, seeds 10000–10099, so both texts play the same quizzes. No log, no
  page list, 4 questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call. vLLM 0.32.0, with
  Gemma from `unsloth/gemma-3-27b-it` as for `gemma_n2`. 1,600 games and about 494,000 model calls.
- **Runner:** `patches/twin-text-runner.patch` restores the hackathon runner's twin sentence, byte for byte. With it,
  the runner reproduces the system prompts of the recorded 2–32-copy aware and twin runs exactly. Runs:
  `scripts/run_twin_grid.sh`. Analysis: `scripts/analyze_twin.py`, committed with this file.

## Registered tests

A test is met when its difference is above 0 and the 95% interval lies above 0. Intervals use 2000 resamples of the
seeds, and every arm of a seed moves together.

- **H1** (Gemma, the reaction to crowding): late holders' posting, twin minus aware, is larger with 64 copies than
  with 4.
- **H2** (Gemma, large N): with 64 copies, collusion per chance in rounds 2 and 4 is higher with the twin text than
  with the aware text.
- **H3** (Gemma, collusion grows with N): collusion per chance over all rounds, twin minus aware, is larger with 64
  copies than with 4.
- **H4** (Gemma, the rule): with 16 copies, the twin-minus-aware gap in late holders' posting is larger with the rule
  than without it. This is the reading in which a copy told that the others are itself does not count posting for them
  as sharing with "any other agent or party".
- **H5** (Qwen3-8B, 64 copies): in rounds 2 and 4, twin minus aware is above 0. The slope of late holders' posting on
  log2(1 + notes seen in round 1), twin minus aware, is also above 0, as with 8–32 copies.

Readings for Gemma:
- **H1 and H3 met:** the twin text's effect grows with swarm size through how copies react to crowded pages.
- **H1 met, H3 not:** it changes that reaction without changing collusion, as in Qwen3-8B up to 32 copies.
- **Neither:** the Qwen3-8B pattern does not carry over to Gemma.

Reported, not registered:
- twin minus aware over all rounds and in rounds 1 and 3, by size;
- early holders' posting;
- Gemma's slopes by size (from 16 copies they rest on few sparse pages);
- the concentration of round-1 titles under each text;
- comparisons with Gemma's unaware-text runs (`gemma_n2`, `gemma_n4` and `gemma_n8` here; 16–64 copies and 16 without
  the rule on the `aaai` branch).
