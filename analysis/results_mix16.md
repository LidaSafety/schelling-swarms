# Mixed swarms at N = 16: scored against the registered forecasts

Forecasts: `predictions_mix16.md`, committed before the runs. Runs: pod I, 18 November (`results/swarm_mix16`, 100 games each, 0 errors; the first attempt failed in every game and was rerun, see the pod I notes). Gemma alone is `results/swarm_gemma_ladder_n16`, the same games and settings run by the ladder of entry 21. Qwen3-8B alone is the paper's main run. Rerun with `python score_mix16.py`.

| Run | Games | Copy-rounds with a round-1 title | CR [95% CI] |
|---|---|---|---|
| 14Q+2G | 100 | 1,519 | 57.0 [53.2, 60.7] |
| 12Q+4G | 100 | 1,524 | 56.2 [52.1, 60.1] |
| 10Q+6G | 100 | 1,533 | 58.9 [55.2, 62.5] |
| 8Q+8G | 100 | 1,548 | 59.8 [56.0, 63.2] |
| 6Q+10G | 100 | 1,554 | 61.4 [57.8, 65.0] |
| 4Q+12G | 100 | 1,566 | 64.5 [61.1, 67.9] |
| 2Q+14G | 100 | 1,583 | 69.5 [65.6, 73.2] |
| Gemma alone | 100 | 1,593 | 73.2 [69.5, 77.1] |
| Qwen3-8B alone | 100 | 1,508 | 58.4 [54.5, 62.2] |

## 1. Title match

M_QQ = 0.303, M_QG = 0.054, M_GG = 0.640 (out/mixtures.json). Registered: all 16 measurements within 0.02 of their forecasts.

| Swarm | x | f(x) | Pooled | Diff | f_N(x) | Same game | Diff | Weighted average |
|---|---|---|---|---|---|---|---|---|
| 14Q+2G | 0.125 | 0.254 | 0.258 | +0.004 | 0.247 | 0.270 | +0.023 | 0.345 |
| 12Q+4G | 0.25 | 0.230 | 0.236 | +0.006 | 0.220 | 0.240 | +0.020 | 0.387 |
| 10Q+6G | 0.375 | 0.233 | 0.243 | +0.009 | 0.220 | 0.237 | +0.016 | 0.429 |
| 8Q+8G | 0.5 | 0.262 | 0.271 | +0.008 | 0.249 | 0.264 | +0.015 | 0.471 |
| 6Q+10G | 0.625 | 0.318 | 0.324 | +0.007 | 0.305 | 0.322 | +0.018 | 0.513 |
| 4Q+12G | 0.75 | 0.399 | 0.406 | +0.007 | 0.388 | 0.407 | +0.018 | 0.555 |
| 2Q+14G | 0.875 | 0.506 | 0.509 | +0.003 | 0.500 | 0.519 | +0.019 | 0.598 |
| Gemma alone | 1 | 0.640 | 0.638 | -0.001 | 0.640 | 0.659 | +0.019 | 0.640 |

14 of 16 within 0.02; largest difference 0.023. **Registered forecast fails.**

Pooled is within 0.009 of f(x) in every row. Same game is above f_N(x) in every row, by 0.015 to 0.023, Gemma alone included (+0.019), where the mixing term is zero: copies of one game match each other a little more often than copies drawn from different games, which f_N(x) does not count. The misses are that, not the mix.

## 2. Collusion below the weighted average

Weighted average (1 − x) × 58.4 + x × 73.2. Registered: sag positive in at least 6 of 7 mixes, interval above zero in at least 4.

| Swarm | x | Weighted average | Measured CR | Sag [95% CI] | Sag at N = 8 (Table 6) |
|---|---|---|---|---|---|
| 14Q+2G | 0.125 | 60.3 | 57.0 | 3.3 [-1.7, 8.3] | – |
| 12Q+4G | 0.25 | 62.1 | 56.2 | 5.9 [1.0, 11.0] | 9.5 |
| 10Q+6G | 0.375 | 64.0 | 58.9 | 5.1 [0.7, 9.6] | – |
| 8Q+8G | 0.5 | 65.8 | 59.8 | 6.1 [1.9, 10.6] | 10.7 |
| 6Q+10G | 0.625 | 67.7 | 61.4 | 6.3 [1.9, 10.8] | – |
| 4Q+12G | 0.75 | 69.5 | 64.5 | 5.1 [0.5, 9.6] | 6.2 |
| 2Q+14G | 0.875 | 71.4 | 69.5 | 1.9 [-3.2, 7.0] | – |

Sag positive in 7 of 7, interval above zero in 5. **Registered forecast holds.**

## 3. For the record: the tabled fuller model

| Swarm | Fuller forecast | Measured | Within 8 points |
|---|---|---|---|
| 12Q+4G | 60.2 | 56.2 | yes |
| 8Q+8G | 63.0 | 59.8 | yes |
| 4Q+12G | 66.6 | 64.5 | yes |
| Gemma alone | 75.9 | 73.2 | yes |

Sag at N = 16 smaller than at N = 8 for the same share: 12Q+4G yes, 8Q+8G yes, 4Q+12G yes.
