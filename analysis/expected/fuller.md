# The fuller model against the simple one, on swarms held out from the count

Every forecast is counted on swarms of N ≤ 8 of the same set and model and applied to the larger swarms; nothing is fitted to them. Collusion rate in percent. Observed: 95% interval from resampling whole games. *N alone*: a straight line in log2 N through the condition's own small swarms. *One curve*: the same line for the Qwen3-8B main runs, used for every condition. *Simple*: the paper's equation 1 with posting forecast, titles and posting counted for round 1 and for rounds 2–4, times one share of finders who use the value. *+ first role*: titles and posting counted for each round on its own. *+ guesses*: use depends on whether the page also shows a seeker's guess. *Fuller*: both. *Fuller, lead*: tried after seeing the held-out swarms, as a check; use with a guess shown depends on whether the true value is posted more often than guesses. Rerun with `python fuller.py`.

## 1. The parameters, counted on N ≤ 8

| Model | Set | π by round (1, 2, 3, 4) | ι by round | u, all finders | u, no guess shown | u, guess shown (finders) |
|---|---|---|---|---|---|---|
| Qwen3-8B | main | 93, 77, 98, 78 | 6, 1, 6, 1 | 98.2 | 99.5 | 67.4 (46) |
| Qwen3-8B | comp | 93, 80, 91, 94 | 3, 2, 6, 4 | 95.4 | 99.3 | 56.6 (159) |
| Qwen3-4B | ours | 95, 93, 98, 93 | 86, 80, 90, 84 | 59.9 | 74.2 | 51.5 (916) |
| Qwen3-14B | ours | 52, 37, 58, 37 | 28, 62, 28, 64 | 74.4 | 88.4 | 33.3 (129) |
| Qwen3-32B | ours | 1, 8, 1, 8 | 12, 2, 12, 2 | 85.7 | 91.7 | 91.7 (4) |
| Gemma 3 27B | gemma | 99, 22, 93, 22 | 2, 1, 3, 2 | 98.6 | 99.1 | 87.5 (56) |

With fewer than 20 finders shown a guess, u with a guess shown is set to u with none.

## 2. Each held-out cell, all rounds

| Condition | N | Observed | N alone | One curve | Simple | + first role | + guesses | Fuller | Fuller, lead |
|---|---|---|---|---|---|---|---|---|---|
| 8B, 2 answers | 16 | 58.4 [54.1, 62.2] | 63.7 | 63.7 | 63.3 | 60.8 | 63.2 | 60.6 | 60.7 |
| 8B, 2 answers | 32 | 70.0 [66.7, 73.1] | 77.7 | 77.7 | 72.9 | 69.1 | 71.8 | 68.0 | 68.3 |
| 8B, 2 answers, rerun | 16 | 61.6 [57.8, 65.2] | 63.7 | 63.7 | 63.3 | 60.8 | 63.2 | 60.6 | 60.7 |
| 8B, 2 answers, rerun | 64 | 78.3 [75.3, 81.0] | 91.6 | 91.6 | 80.9 | 76.5 | 77.8 | 73.6 | 74.2 |
| 8B, 1 answer | 16 | 66.4 [63.8, 68.8] | 67.3 | 63.7 | 65.9 | 65.8 | 64.0 | 63.9 | 65.5 |
| 8B, 1 answer | 32 | 76.2 [74.2, 78.3] | 82.7 | 77.7 | 74.5 | 74.3 | 68.3 | 68.2 | 73.1 |
| 3-4B | 16 | 38.4 [34.4, 42.4] | 38.8 | 63.7 | 47.8 | 45.3 | 42.7 | 40.3 | 40.0 |
| 3-4B | 32 | 41.5 [38.1, 44.8] | 43.9 | 77.7 | 51.1 | 48.6 | 44.7 | 42.3 | 41.9 |
| 3-14B | 16 | 22.0 [19.0, 25.0] | 21.6 | 63.7 | 33.9 | 30.8 | 28.4 | 26.3 | 28.5 |
| 3-14B | 32 | 29.8 [27.0, 32.9] | 26.9 | 77.7 | 49.1 | 44.4 | 32.5 | 30.6 | 37.8 |
| 3-32B | 16 | 3.1 [1.7, 4.7] | 1.5 | 63.7 | 3.6 | 2.9 | 3.8 | 3.1 | 3.1 |
| 3-32B | 32 | 8.6 [6.4, 11.2] | 1.9 | 77.7 | 6.7 | 5.5 | 7.1 | 5.8 | 5.8 |

## 3. The two halves of each cell

| Condition | N | Half | Observed | Simple | + first role | + guesses | Fuller | Fuller, lead |
|---|---|---|---|---|---|---|---|---|
| 8B, 2 answers | 16 | rounds 1 and 3 | 63.4 [59.1, 67.8] | 64.4 | 66.9 | 64.0 | 66.1 | 66.3 |
| 8B, 2 answers | 16 | rounds 2 and 4 | 53.4 [48.4, 58.2] | 62.2 | 54.6 | 62.3 | 55.1 | 55.2 |
| 8B, 2 answers | 32 | rounds 1 and 3 | 70.1 [65.8, 74.4] | 73.4 | 74.6 | 71.8 | 72.1 | 72.6 |
| 8B, 2 answers | 32 | rounds 2 and 4 | 69.9 [65.7, 73.9] | 72.4 | 63.6 | 71.9 | 63.9 | 64.0 |
| 8B, 2 answers, rerun | 16 | rounds 1 and 3 | 65.4 [61.1, 69.8] | 64.4 | 66.9 | 64.0 | 66.1 | 66.3 |
| 8B, 2 answers, rerun | 16 | rounds 2 and 4 | 57.7 [52.6, 62.5] | 62.2 | 54.6 | 62.3 | 55.1 | 55.2 |
| 8B, 2 answers, rerun | 64 | rounds 1 and 3 | 76.8 [73.0, 80.7] | 80.7 | 80.5 | 76.6 | 74.8 | 75.8 |
| 8B, 2 answers, rerun | 64 | rounds 2 and 4 | 79.7 [76.5, 82.7] | 81.2 | 72.5 | 79.0 | 72.3 | 72.5 |
| 8B, 1 answer | 16 | rounds 1 and 3 | 65.7 [62.4, 68.9] | 65.8 | 66.3 | 64.0 | 63.8 | 65.5 |
| 8B, 1 answer | 16 | rounds 2 and 4 | 67.1 [64.2, 70.1] | 66.0 | 65.3 | 64.1 | 64.0 | 65.4 |
| 8B, 1 answer | 32 | rounds 1 and 3 | 75.3 [72.6, 77.9] | 74.6 | 75.1 | 68.5 | 67.6 | 73.1 |
| 8B, 1 answer | 32 | rounds 2 and 4 | 77.2 [75.1, 79.3] | 74.4 | 73.6 | 68.0 | 68.8 | 73.0 |
| 3-4B | 16 | rounds 1 and 3 | 50.6 [44.7, 56.4] | 47.3 | 46.4 | 42.0 | 40.8 | 40.4 |
| 3-4B | 16 | rounds 2 and 4 | 26.3 [21.8, 31.3] | 48.4 | 44.1 | 43.4 | 39.7 | 39.5 |
| 3-4B | 32 | rounds 1 and 3 | 52.6 [47.1, 58.3] | 51.2 | 51.5 | 44.6 | 44.6 | 43.8 |
| 3-4B | 32 | rounds 2 and 4 | 30.5 [25.8, 35.5] | 51.0 | 45.7 | 44.7 | 39.9 | 39.9 |
| 3-14B | 16 | rounds 1 and 3 | 28.6 [23.9, 33.8] | 33.4 | 34.0 | 28.9 | 31.3 | 33.6 |
| 3-14B | 16 | rounds 2 and 4 | 15.3 [11.4, 19.3] | 34.3 | 27.7 | 28.0 | 21.4 | 23.4 |
| 3-14B | 32 | rounds 1 and 3 | 37.1 [32.6, 41.8] | 48.3 | 48.6 | 33.7 | 37.5 | 45.7 |
| 3-14B | 32 | rounds 2 and 4 | 22.6 [18.6, 27.3] | 49.8 | 40.2 | 31.3 | 23.6 | 30.0 |
| 3-32B | 16 | rounds 1 and 3 | 0.9 [0.1, 2.2] | 2.6 | 0.8 | 2.8 | 0.9 | 0.9 |
| 3-32B | 16 | rounds 2 and 4 | 5.3 [2.8, 8.3] | 4.5 | 5.0 | 4.8 | 5.4 | 5.4 |
| 3-32B | 32 | rounds 1 and 3 | 1.2 [0.5, 2.1] | 5.0 | 1.6 | 5.3 | 1.7 | 1.7 |
| 3-32B | 32 | rounds 2 and 4 | 16.1 [11.6, 20.8] | 8.4 | 9.4 | 9.0 | 10.0 | 10.0 |

## 4. Errors by group

Error: forecast minus observed, in points.

| Cells | Forecast | Median error | Largest error | Mean signed error | Inside the observed interval |
|---|---|---|---|---|---|
| Qwen3-8B, all rounds (6 cells) | N alone | 5.9 | 13.4 | +6.0 | 2 of 6 |
| Qwen3-8B, all rounds (6 cells) | One curve | 4.0 | 13.4 | +4.5 | 2 of 6 |
| Qwen3-8B, all rounds (6 cells) | Simple | 2.2 | 4.9 | +1.7 | 5 of 6 |
| Qwen3-8B, all rounds (6 cells) | + first role | 1.3 | 2.4 | -0.6 | 6 of 6 |
| Qwen3-8B, all rounds (6 cells) | + guesses | 2.1 | 8.0 | -0.4 | 4 of 6 |
| Qwen3-8B, all rounds (6 cells) | Fuller | 2.4 | 8.0 | -2.6 | 4 of 6 |
| Qwen3-8B, all rounds (6 cells) | Fuller, lead | 2.0 | 4.1 | -1.4 | 4 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | N alone | 2.0 | 6.8 | -1.5 | 3 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | One curve | 44.8 | 69.0 | +46.8 | 0 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | Simple | 9.5 | 19.2 | +8.1 | 2 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | + first role | 6.9 | 14.5 | +5.7 | 1 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | + guesses | 2.9 | 6.5 | +2.6 | 4 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | Fuller | 1.3 | 4.4 | +0.8 | 4 of 6 |
| 4B, 14B, 32B, all rounds (6 cells) | Fuller, lead | 2.2 | 8.0 | +2.3 | 3 of 6 |
| Every cell, all rounds (12) | N alone | 2.6 | 13.4 | +2.2 | 5 of 12 |
| Every cell, all rounds (12) | One curve | 19.3 | 69.0 | +25.7 | 2 of 12 |
| Every cell, all rounds (12) | Simple | 2.8 | 19.2 | +4.9 | 7 of 12 |
| Every cell, all rounds (12) | + first role | 2.1 | 14.5 | +2.5 | 7 of 12 |
| Every cell, all rounds (12) | + guesses | 2.5 | 8.0 | +1.1 | 8 of 12 |
| Every cell, all rounds (12) | Fuller | 2.1 | 8.0 | -0.9 | 8 of 12 |
| Every cell, all rounds (12) | Fuller, lead | 2.0 | 8.0 | +0.4 | 7 of 12 |
| Every cell, by half (24) | Simple | 3.3 | 27.2 | +4.9 | 13 of 24 |
| Every cell, by half (24) | + first role | 3.6 | 17.8 | +2.5 | 13 of 24 |
| Every cell, by half (24) | + guesses | 3.7 | 17.1 | +1.1 | 11 of 24 |
| Every cell, by half (24) | Fuller | 2.7 | 13.4 | -0.9 | 13 of 24 |
| Every cell, by half (24) | Fuller, lead | 3.5 | 13.2 | +0.4 | 13 of 24 |

## 5. Mixed swarms, from the pure swarms

Each model's counts come from its own pure swarms at N ≤ 8 (Qwen3-8B from the main runs, Gemma 3 27B from its own). *Pure swarms' average*: the paper's dashed line, the pure swarms' rates at the same N weighted by model share.

| Swarm | Observed | One curve | Pure swarms' average | Simple | + first role | + guesses | Fuller | Fuller, lead |
|---|---|---|---|---|---|---|---|---|
| 2 Qwen3-8B + 2 Gemma 3 27B | 29.5 [24.9, 34.5] | 35.8 | 44.4 | 31.0 | 30.7 | 31.2 | 30.9 | 30.9 |
| 2 Qwen3-8B + 6 Gemma 3 27B | 52.4 [47.4, 57.1] | 49.7 | 58.6 | 58.3 | 53.7 | 58.4 | 53.8 | 53.8 |
| 4 Qwen3-8B + 4 Gemma 3 27B | 44.8 [40.2, 49.3] | 49.7 | 55.5 | 49.3 | 47.2 | 49.5 | 47.4 | 47.4 |
| 6 Qwen3-8B + 2 Gemma 3 27B | 42.9 [38.6, 47.2] | 49.7 | 52.3 | 46.0 | 44.9 | 46.3 | 45.2 | 45.2 |

| Forecast | Median error | Largest error | Inside |
|---|---|---|---|
| One curve | 5.6 | 6.9 | 1 of 4 |
| Pure swarms average | 10.1 | 14.9 | 0 of 4 |
| Simple | 3.8 | 5.9 | 3 of 4 |
| + first role | 1.7 | 2.5 | 4 of 4 |
| + guesses | 4.1 | 6.0 | 2 of 4 |
| Fuller | 1.9 | 2.7 | 4 of 4 |
| Fuller, lead | 1.9 | 2.7 | 4 of 4 |

## 6. Forecasts for swarms not yet run (draft; register before any run)

The same counts on N ≤ 8, applied to swarms that have not been run. Each block of N/2 copies holds one half of the answers; in a mix each block has the swarm's share of each model. Collusion rate in percent, over all seekers, then the two halves.

| Swarm | N | Simple | + first role | + guesses | Fuller | Fuller, rounds 1 and 3 | Fuller, rounds 2 and 4 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 64 | 80.9 | 76.5 | 77.8 | 73.6 | 74.8 | 72.3 |
| Qwen3-8B | 128 | 85.6 | 81.5 | 79.1 | 75.7 | 73.7 | 77.8 |
| Qwen3-8B | 256 | 87.2 | 83.6 | 76.0 | 74.2 | 69.7 | 78.8 |
| Qwen3-4B | 64 | 53.3 | 49.9 | 46.5 | 43.2 | 46.0 | 40.5 |
| Qwen3-4B | 128 | 55.4 | 51.0 | 48.2 | 44.2 | 46.6 | 41.8 |
| Qwen3-14B | 64 | 60.7 | 55.2 | 32.9 | 31.4 | 37.6 | 25.1 |
| Qwen3-14B | 128 | 66.8 | 61.9 | 32.7 | 31.0 | 34.3 | 27.7 |
| Qwen3-32B | 64 | 11.9 | 9.7 | 12.8 | 10.4 | 3.2 | 17.6 |
| Qwen3-32B | 128 | 19.7 | 15.8 | 21.1 | 16.9 | 5.5 | 28.2 |
| Gemma 3 27B | 16 | 89.3 | 76.6 | 88.6 | 75.9 | 97.1 | 54.7 |
| Gemma 3 27B | 32 | 96.1 | 82.4 | 94.4 | 80.6 | 95.7 | 65.6 |

Mixed swarms of Qwen3-8B and Gemma 3 27B at N = 16 and 32, with the weighted average of the forecast pure swarms (the paper's dashed line) and the sag, forecast minus average:

| Swarm | N | Fuller | Weighted average of the pure forecasts | Sag |
|---|---|---|---|---|
| 6 Qwen3-8B + 2 Gemma 3 27B | 8 | 45.2 | 54.0 | -8.8 |
| 4 Qwen3-8B + 4 Gemma 3 27B | 8 | 47.4 | 58.2 | -10.7 |
| 2 Qwen3-8B + 6 Gemma 3 27B | 8 | 53.8 | 62.3 | -8.6 |
| 12 Qwen3-8B + 4 Gemma 3 27B | 16 | 60.2 | 64.5 | -4.3 |
| 8 Qwen3-8B + 8 Gemma 3 27B | 16 | 63.0 | 68.3 | -5.2 |
| 4 Qwen3-8B + 12 Gemma 3 27B | 16 | 66.6 | 72.1 | -5.5 |
| 24 Qwen3-8B + 8 Gemma 3 27B | 32 | 70.6 | 71.2 | -0.6 |
| 16 Qwen3-8B + 16 Gemma 3 27B | 32 | 73.9 | 74.3 | -0.4 |
| 8 Qwen3-8B + 24 Gemma 3 27B | 32 | 76.1 | 77.5 | -1.4 |

## 7. The rule, by round (N = 16)

The paper's Section 4.5 removed the rule sentence and kept everything else, paired by quiz. The registered expectation was that 4B and 8B would move by less than 5 points. 8B rose by 17. Here each step is split by round: in rounds 1 and 3 the holders began as holders, in rounds 2 and 4 as seekers. Title match: Σ_t r_t w_t over the seekers and posting holders of those rounds. Percent, except the title match.

| Model | Rule | Rounds | Holders who post | Seekers who post a guess | Title match | Found | Found → used | CR |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | yes | 1 and 3 | 97.6 | 86.2 | 0.30 | 75.3 | 67.1 | 50.6 |
| Qwen3-4B | yes | 2 and 4 | 91.5 | 86.3 | 0.31 | 65.0 | 40.5 | 26.3 |
| Qwen3-4B | no | 1 and 3 | 91.4 | 78.5 | 0.37 | 85.6 | 72.6 | 62.2 |
| Qwen3-4B | no | 2 and 4 | 84.9 | 83.4 | 0.40 | 69.3 | 25.6 | 17.9 |
| Qwen3-8B | yes | 1 and 3 | 95.8 | 5.4 | 0.26 | 66.1 | 99.1 | 65.4 |
| Qwen3-8B | yes | 2 and 4 | 82.1 | 0.4 | 0.25 | 57.8 | 99.8 | 57.7 |
| Qwen3-8B | no | 1 and 3 | 96.1 | 8.0 | 0.52 | 80.1 | 100.0 | 80.1 |
| Qwen3-8B | no | 2 and 4 | 98.4 | 0.7 | 0.52 | 76.6 | 100.0 | 76.6 |
| Qwen3-14B | yes | 1 and 3 | 55.9 | 26.8 | 0.15 | 39.3 | 72.8 | 28.6 |
| Qwen3-14B | yes | 2 and 4 | 28.9 | 54.2 | 0.18 | 25.3 | 60.5 | 15.3 |
| Qwen3-14B | no | 1 and 3 | 47.8 | 48.6 | 0.11 | 30.4 | 39.3 | 11.9 |
| Qwen3-14B | no | 2 and 4 | 74.1 | 68.4 | 0.15 | 51.4 | 54.6 | 28.1 |
| Qwen3-32B | yes | 1 and 3 | 1.1 | 9.3 | 0.15 | 0.9 | 100.0 | 0.9 |
| Qwen3-32B | yes | 2 and 4 | 5.8 | 3.6 | 0.14 | 5.9 | 90.4 | 5.3 |
| Qwen3-32B | no | 1 and 3 | 18.8 | 30.1 | 0.11 | 15.1 | 85.1 | 12.8 |
| Qwen3-32B | no | 2 and 4 | 25.0 | 18.4 | 0.18 | 32.4 | 85.5 | 27.8 |
