# Forecasts for the swarms mixing Qwen3-4B, Qwen3-8B and Gemma 3 27B

Written and committed before these runs start (pod J, 22 November). They measure the paper's Figure 14 (`fig:simplex`, Appendix F), whose mixes with Qwen3-4B were predictions only. Runs: benchmark branch `mixed-3way`, `configs/experiments.yaml` entry 26, run from `results/logs/podJ/pod/mix3.sh`: tag `mix3`, seed 1 (games from 10000), unaware, rule in force, no listing, no log, 4 questions, 2 answers per copy (fixed blocks), Qwen3 with thinking off, the three models served together on one H200. Seats in the order 4B, 8B, Gemma; every model has an even number of copies, so it holds answers in both halves of the quiz. 22 mixes, 21 of 16 copies and the even three-way mix 4:4:4 with 12 (16 copies cannot be split evenly in three). Rerun with `python predict_mix3.py <hackathon clone>/scaling` (it will not overwrite this file).

Four of the mixes (6:8:2, 8:8:0, 4:4:8, 8:0:8) were registered first by ykazemi in the hackathon repo (branch `scaling`, acba44a, `scaling/results/predictions_mix3.md`) and are run with that file's game counts. Its criteria (a)-(d) stand as written there and are scored separately; the CR forecasts below use its script (`scaling/scripts/predict_mix3.py`, imported unchanged at 244c115) and reproduce its four numbers exactly.

Notation (copies of 4B : 8B : Gemma). Pure swarms of 16 copies, measured: 4B 38.2 (100 games), 8B 58.4 (100 games), Gemma 73.2 (100 games) (4B: `results/swarm4round/swarm_ref_4b`; 8B: the paper's main run; Gemma: `results/swarm_gemma_ladder_n16`).

| Block | Swarm | N | Games | `--mix` |
|---|---|---|---|---|
| registered | 6:8:2 | 16 | 200 | `qwen3-4b:6,qwen3-8b:8,gemma-3-27b:2` |
| registered | 8:8:0 | 16 | 200 | `qwen3-4b:8,qwen3-8b:8` |
| registered | 4:4:8 | 16 | 100 | `qwen3-4b:4,qwen3-8b:4,gemma-3-27b:8` |
| registered | 8:0:8 | 16 | 100 | `qwen3-4b:8,gemma-3-27b:8` |
| inside | 6:6:4 | 16 | 100 | `qwen3-4b:6,qwen3-8b:6,gemma-3-27b:4` |
| inside | 8:4:4 | 16 | 100 | `qwen3-4b:8,qwen3-8b:4,gemma-3-27b:4` |
| inside | 4:8:4 | 16 | 100 | `qwen3-4b:4,qwen3-8b:8,gemma-3-27b:4` |
| inside | 6:4:6 | 16 | 100 | `qwen3-4b:6,qwen3-8b:4,gemma-3-27b:6` |
| inside | 4:6:6 | 16 | 100 | `qwen3-4b:4,qwen3-8b:6,gemma-3-27b:6` |
| inside | 4:4:4 | 12 | 100 | `qwen3-4b:4,qwen3-8b:4,gemma-3-27b:4` |
| edge 4B+8B | 14:2:0 | 16 | 100 | `qwen3-4b:14,qwen3-8b:2` |
| edge 4B+8B | 12:4:0 | 16 | 100 | `qwen3-4b:12,qwen3-8b:4` |
| edge 4B+8B | 10:6:0 | 16 | 100 | `qwen3-4b:10,qwen3-8b:6` |
| edge 4B+8B | 6:10:0 | 16 | 100 | `qwen3-4b:6,qwen3-8b:10` |
| edge 4B+8B | 4:12:0 | 16 | 100 | `qwen3-4b:4,qwen3-8b:12` |
| edge 4B+8B | 2:14:0 | 16 | 100 | `qwen3-4b:2,qwen3-8b:14` |
| edge 4B+Gemma | 14:0:2 | 16 | 100 | `qwen3-4b:14,gemma-3-27b:2` |
| edge 4B+Gemma | 12:0:4 | 16 | 100 | `qwen3-4b:12,gemma-3-27b:4` |
| edge 4B+Gemma | 10:0:6 | 16 | 100 | `qwen3-4b:10,gemma-3-27b:6` |
| edge 4B+Gemma | 6:0:10 | 16 | 100 | `qwen3-4b:6,gemma-3-27b:10` |
| edge 4B+Gemma | 4:0:12 | 16 | 100 | `qwen3-4b:4,gemma-3-27b:12` |
| edge 4B+Gemma | 2:0:14 | 16 | 100 | `qwen3-4b:2,gemma-3-27b:14` |

## 1. Title match (the paper's Appendix F)

From the paper's matrix M (round-1 writes, pure swarms of N ≤ 8; 4B–4B 0.335, 8B–8B 0.303, Gemma–Gemma 0.640, 4B–8B 0.027, 4B–Gemma 0.047, 8B–Gemma 0.054): f = Σ x_m x_m' M_mm' for any two round-1 titles of the mix (*pooled*), f_N for two different copies of one game (*same game*), and the weighted average of the pure swarms. Measured as in `score_mix16.py` (round-1 titles, state replaced by a placeholder).

| Swarm | f, pooled | f_N, same game | Weighted average |
|---|---|---|---|
| 6:8:2 | **0.154** | 0.141 | 0.357 |
| 8:8:0 | **0.173** | 0.163 | 0.319 |
| 4:4:8 | **0.229** | 0.212 | 0.480 |
| 8:0:8 | **0.267** | 0.253 | 0.488 |
| 6:6:4 | **0.156** | 0.140 | 0.399 |
| 8:4:4 | **0.168** | 0.152 | 0.403 |
| 4:8:4 | **0.163** | 0.147 | 0.395 |
| 6:4:6 | **0.184** | 0.167 | 0.441 |
| 4:6:6 | **0.183** | 0.166 | 0.437 |
| 4:4:4 | **0.170** | 0.147 | 0.426 |
| 14:2:0 | **0.267** | 0.263 | 0.331 |
| 12:4:0 | **0.217** | 0.210 | 0.327 |
| 10:6:0 | **0.186** | 0.177 | 0.323 |
| 6:10:0 | **0.178** | 0.169 | 0.315 |
| 4:12:0 | **0.201** | 0.194 | 0.311 |
| 2:14:0 | **0.243** | 0.239 | 0.307 |
| 14:0:2 | **0.277** | 0.270 | 0.373 |
| 12:0:4 | **0.246** | 0.235 | 0.411 |
| 10:0:6 | **0.243** | 0.229 | 0.449 |
| 6:0:10 | **0.319** | 0.305 | 0.526 |
| 4:0:12 | **0.399** | 0.388 | 0.564 |
| 2:0:14 | **0.506** | 0.499 | 0.602 |

**Registered:**
- (1a) Every one of the 22 measured pooled title matches is within 0.02 of f.
- (1b) The lowest measured pooled title match of the 21 mixes of 16 copies is a three-model mix (forecast: 6:8:2 at 0.154, against 0.173 for the lowest two-model mix, 8:8:0).

Reported, not registered: same game. In the 8B + Gemma mixes of 16 copies it ran 0.015 to 0.023 above f_N, and Gemma alone 0.019 above, so a similar offset is expected.

## 2. Collusion rate (the prediction from pairs, with wrong guesses)

ykazemi's extension of the paper's prediction from pairs to several models: a seeker of model a sees R = Σ_m Bin(h_m, π_am) true values and W = Σ_m Bin(g_m, ω_am) wrong guesses, each rate measured between that model's seekers and model m's holders (or non-holders) in different base pairs; it uses a true value at its pairs' uptake u_a when no guess is beside it, and in proportion R / (R + W) when one is. Base pairs: 100 each, uptake 4B 0.814, 8B 1.000, Gemma 1.000. Intervals: 200 resamples of the base pairs. For the record, the same prediction without guesses (`title_law.py`, which the paper used for its two-model mixes).

| Swarm | CR predicted, % [95%] | Without guesses | Weighted average of measured pure | Sag forecast |
|---|---|---|---|---|
| 6:8:2 | **45.4** [41.4, 48.8] | 50.9 | 52.7 | 7.3 |
| 8:8:0 | **44.0** [39.5, 47.6] | 52.9 | 48.3 | 4.3 |
| 4:4:8 | **50.5** [47.3, 53.3] | 53.5 | 60.8 | 10.3 |
| 8:0:8 | **49.4** [45.5, 52.2] | 58.1 | 55.8 | 6.3 |
| 6:6:4 | **45.4** [41.9, 48.4] | 50.7 | 54.6 | 9.2 |
| 8:4:4 | **42.3** [39.0, 45.2] | 50.8 | 52.0 | 9.7 |
| 4:8:4 | **48.7** [45.1, 52.0] | 51.5 | 57.1 | 8.4 |
| 6:4:6 | **46.2** [43.0, 48.7] | 51.7 | 56.4 | 10.2 |
| 4:6:6 | **49.3** [46.0, 52.1] | 52.2 | 58.9 | 9.6 |
| 4:4:4 | **42.1** [38.9, 44.6] | 44.8 | – | – |
| 14:2:0 | **36.0** [30.7, 39.7] | 55.2 | 40.8 | 4.8 |
| 12:4:0 | **37.7** [33.5, 41.3] | 53.4 | 43.3 | 5.6 |
| 10:6:0 | **40.7** [35.9, 44.4] | 52.9 | 45.8 | 5.1 |
| 6:10:0 | **47.4** [42.8, 51.6] | 53.1 | 50.8 | 3.5 |
| 4:12:0 | **50.5** [45.3, 55.4] | 53.6 | 53.4 | 2.9 |
| 2:14:0 | **53.4** [47.6, 59.1] | 54.6 | 55.9 | 2.5 |
| 14:0:2 | **38.4** [32.8, 42.0] | 57.4 | 42.6 | 4.2 |
| 12:0:4 | **41.4** [36.8, 44.5] | 56.8 | 47.0 | 5.6 |
| 10:0:6 | **45.2** [41.2, 48.1] | 57.2 | 51.4 | 6.1 |
| 6:0:10 | **53.9** [49.8, 57.0] | 59.6 | 60.1 | 6.2 |
| 4:0:12 | **58.7** [53.6, 62.0] | 62.1 | 64.5 | 5.8 |
| 2:0:14 | **64.2** [58.3, 68.1] | 66.3 | 68.9 | 4.7 |

**Registered:**
- (2a) The measured CR lies inside the 95% interval above in at least 18 of the 22 mixes (80%).
- (2b) The median |measured − predicted| over the 22 mixes is at most 3 points (the paper: median 1.5 for swarms of 4–32 copies under the rule of their pairs; the 11 two-model mixes within 2.6).

## 3. Sag: mixes collude less than the weighted average of their pure swarms (the paper's Section 4.3)

Sag = weighted average of the measured pure swarms of 16 copies (above) less the mix's measured CR, with a 95% interval from resampling games, the four runs independently. Only the 21 mixes of 16 copies (no pure swarm of 12 copies has been run). The forecast in the table of section 2 subtracts the CR predicted from pairs. By the paper's equation (gap = Σ x_m x_m' D_mm'), mixes of three models lose more title matches than mixes of two, so they should sag more: the forecast means are 9.2 points for the 7 three-model mixes and 4.8 for the 14 two-model mixes.

**Registered:**
- (3a) The measured sag is positive in at least 17 of the 21 mixes of 16 copies.
- (3b) The mean measured sag of the 7 three-model mixes is larger than that of the 14 two-model mixes.

Reported, not registered: the sag along each new edge against pod I's 8B + Gemma edge, and the 4:4:4 mix of 12 against the prediction from pairs for pure swarms of 12 (4B 37.0, 8B 52.8, Gemma 68.9; weighted average 52.9).

Scoring: `score_mix3.py`, to be written after the runs, reading `results/swarm_mix3/`. ykazemi's criteria (a)–(d) are scored on the same runs.
