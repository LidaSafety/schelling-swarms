# Forecasts for the three-model swarms at N = 16

Written before these runs start, and committed before the first run. Runs: four swarms of 16 copies mixing Qwen3-4B,
Qwen3-8B and Gemma 3 27B, on the fixed prompt (unaware, rule in force, no page list, `--log none`), 4 questions,
2 answers per copy (`--f-inv 2`, fixed blocks), temperature 1, 250 tokens per call, Qwen3 with thinking off, the
models served as for their pairs. Seeds from 10000, as every earlier run. Each model has an even number of copies, so
it holds answers in both halves of the swarm.

| Swarm (4B : 8B : Gemma) | `--mix` | Games |
|---|---|---|
| 8 : 8 : 0 | `qwen3-4b:8,qwen3-8b:8` | 200 |
| 8 : 0 : 8 | `qwen3-4b:8,gemma-3-27b:8` | 100 |
| 6 : 8 : 2 | `qwen3-4b:6,qwen3-8b:8,gemma-3-27b:2` | 200 |
| 4 : 4 : 8 | `qwen3-4b:4,qwen3-8b:4,gemma-3-27b:8` | 100 |

What they test: the two edges of the paper's Figure 14 (`fig:simplex`) that no run has measured (4B with 8B, 4B with
Gemma); the predicted minimum, where a three-model mix matches titles less often than any two-model mix; a point higher
inside the triangle; and the guess term of the prediction from pairs, which matters with Qwen3-4B.

Base: each model's 100 pairs on the same prompt (Qwen3-4B `wiki_game/data/swarm4round/swarm_ref_4b`, Qwen3-8B
`results/swarm_main_n2`, Gemma `results/swarm_gemma_n2`). Uptake in those pairs: 4B 0.814, 8B 1.0, Gemma 1.0.
Script: `scripts/predict_mix3.py` (numbers in `results/predictions_mix3.json`); 95% intervals from 200 resamples of
the base pairs.

## 1. Title match (the paper's Appendix F)

From the paper's matrix M (round-1 writes, pure swarms of N ≤ 8): f = sum over model pairs of x_m x_m' M_mm', pooled;
f_N for two distinct copies of one game; the weighted average of the pure swarms for comparison.

| Swarm | f, pooled | f_N, same game | Weighted average |
|---|---|---|---|
| 8 : 8 : 0 | **0.173** | 0.163 | 0.319 |
| 8 : 0 : 8 | **0.267** | 0.253 | 0.488 |
| 6 : 8 : 2 | **0.154** | 0.141 | 0.357 |
| 4 : 4 : 8 | **0.229** | 0.212 | 0.480 |

**Registered:**
- (a) Every measured pooled title match is within 0.02 of f.
- (b) The measured pooled title match of 6 : 8 : 2 is below that of 8 : 8 : 0, as the paper's claim that a three-model
  mix can match less often than any two-model mix requires.

Reported, not registered: the same-game values. In the 16-copy mixes they ran 0.015 to 0.023 above f_N, and Gemma
alone 0.019 above, so a similar offset is expected here.

## 2. Collusion rate (the prediction from pairs, with guesses)

The paper's prediction from pairs, extended to several models: a seeker of model a sees R = sum_m Bin(h_m, pi_am)
true values and W = sum_m Bin(g_m, omega_am) wrong guesses, each rate measured between that model's seekers and the
other model's holders (or non-holders) in different pairs; it uses a true value at rate u_a when no guess is beside
it, and in proportion R / (R + W) when one is.

| Swarm | CR predicted, % [95%] | Found values beside a guess, % [95%] |
|---|---|---|
| 8 : 8 : 0 | **44.0** [39.5, 47.6] | 45.7 [39.3, 50.7] |
| 8 : 0 : 8 | **49.4** [45.5, 52.2] | 42.3 [36.9, 45.8] |
| 6 : 8 : 2 | **45.4** [41.4, 48.8] | 29.8 [25.1, 34.2] |
| 4 : 4 : 8 | **50.5** [47.3, 53.3] | 15.2 [11.8, 19.0] |

**Registered:**
- (c) The measured CR lies inside the 95% interval in at least 3 of the 4 swarms.
- (d) In the three swarms with 6 or more Qwen3-4B copies (8 : 8 : 0, 8 : 0 : 8, 6 : 8 : 2), the measured CR is closer
  to this forecast than to the forecast without guesses (section 3): below 48.45, 53.78 and 48.12.

Reported, not registered: the share of found values beside a guess.

## 3. For the record: without guesses

The title law without guesses, which the paper used for its two-model mixes (`scripts/title_law.py`): 8 : 8 : 0 52.9,
8 : 0 : 8 58.1, 6 : 8 : 2 50.9, 4 : 4 : 8 53.5. The guesses come mostly from Qwen3-4B, whose seekers guess in 86% of
their write steps.

## Checks before writing

- Pure swarms of 16 copies, from the same code: 4B 37.3, 8B 56.2, Gemma 72.3, as `scripts/law_wrong.py` and the paper.
- The 11 measured mixes of Qwen3-8B and Gemma (4, 8 and 16 copies): all inside the interval of this prediction; the
  measured rate is between 0.2 points below and 3.8 points above it (median 1.4 above), the largest for 2 Qwen + 14
  Gemma.

## Scoring

After the runs, the same script reads the measured swarms and scores (a) to (d); the base pairs and M stay as above.
