# Forecasts for the mixed swarms at N = 16

Written and committed before these runs start. Runs: Gemma 3 27B alone at N = 16, and Gemma shares in eighths, 14 + 2, 12 + 4, ..., 2 + 14 copies of Qwen3-8B + Gemma 3 27B (benchmark branch `mixed-n16`, `configs/experiments.yaml` entry 20, run from `results/logs/podH/pod/mix16.sh`): tag `mix16`, seed 1 (games 10000–10099), 100 games each, unaware, rule in force, no listing, no log, 2 answers per copy, each model holding both halves of the answers. Qwen3-8B alone at N = 16 is the paper's main run (58.4%). Rerun with `python predict_mix16.py` (it will not overwrite this file). A first version with three mixes (Gemma shares 1/4, 1/2, 3/4) was committed earlier on 18 November and replaced before any run; its forecasts for those three mixes are unchanged here.

## 1. Title match (the paper's Appendix F)

One round-1 title distribution per model, counted on pure swarms of N ≤ 8 (`mixtures.md`): M_QQ = 0.303, M_QG = 0.054, M_GG = 0.640, D = 0.835. For Gemma share x, f(x) = (1 − x)² M_QQ + 2x(1 − x) M_QG + x² M_GG, the gap below the weighted average is x(1 − x) D, and two different copies of one game of N copies match with f_N(x) = f(x) − x(1 − x) D / (N − 1). Measured as in `mixtures.py` section 2: *pooled*, any two round-1 titles of the condition (a title not paired with itself); *same game*, the round-1 titles of two different copies of one game.

| Swarm | Gemma share x | f(x), pooled | f_N(x), same game | Weighted average |
|---|---|---|---|---|
| 14Q+2G | 0.125 | **0.254** | **0.247** | 0.345 |
| 12Q+4G | 0.25 | **0.230** | **0.220** | 0.387 |
| 10Q+6G | 0.375 | **0.233** | **0.220** | 0.429 |
| 8Q+8G | 0.5 | **0.262** | **0.249** | 0.471 |
| 6Q+10G | 0.625 | **0.318** | **0.305** | 0.513 |
| 4Q+12G | 0.75 | **0.399** | **0.388** | 0.555 |
| 2Q+14G | 0.875 | **0.506** | **0.500** | 0.598 |
| Gemma alone | 1 | **0.640** | **0.640** | 0.640 |

**Registered:** every one of the 16 measured title matches above (pooled and same game, 7 mixes and Gemma alone) is within 0.02 of its forecast. At N ≤ 8 the largest difference was 0.015 (`mixtures.md` section 2).

## 2. Collusion below the weighted average (the paper's Section 4.3)

The weighted average at N = 16 is (1 − x) × 58.4 + x × Gemma's measured CR at N = 16 from this session. Sag: weighted average less the mix's CR, with a 95% interval from resampling games (the three runs resampled independently, as in `mixtures.md`).

**Registered:** the measured sag is positive in at least 6 of the 7 mixes, and its 95% interval lies above zero in at least 4 of them. The two mixes with only 2 copies of one model (14 + 2, 2 + 14) sit closest to a pure swarm, where the sag is smallest.

## 3. For the record: the tabled fuller model

The model of the paper's second version, which the AAAI version does not use. Its forecasts were drafted in `fuller.md` section 6 on 17 November, for three of the seven mixes only; they are copied here unchanged. Collusion rate in %.

| Swarm | Fuller | Weighted average of its pure forecasts | Sag |
|---|---|---|---|
| 12Q+4G | 60.2 | 64.5 | 4.3 |
| 8Q+8G | 63.0 | 68.3 | 5.2 |
| 4Q+12G | 66.6 | 72.1 | 5.5 |
| Gemma alone | 75.9 (simple model 89.3) | | |

Two checks, recorded but not counted as a test of the paper: (a) each forecast within 8 points of the measured CR (the accuracy bound of `predictions_pod5.md`); (b) the measured sag at N = 16 is smaller than the measured sag at N = 8 for the same share (9.5 for x = 1/4, 10.7 for x = 1/2, 6.2 for x = 3/4), as the model forecasts (4–6 points).

Scoring: a `score_mix16.py` is to be written after the runs, reading `results/swarm_mix16/`.
