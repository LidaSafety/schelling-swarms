# The find term from titles alone (4-round game)

find(K) = Σ_t r_t (1 − (1 − w_t)^K): r_t is the share of seekers that read title t, w_t the share of posting holders whose posts of the value include it, K the number of other holders that posted it. Nothing is fitted: r and w are counted on swarms of N ≤ 8 and the larger swarms are predictions. With K = 1 this is the title match of the hackathon report. In sections 3 to 5 K is counted in each target game, so the coverage term predicts finding given posting; section 6 forecasts K too. Unaware text, rule in force. Rerun with `python coverage.py`.

## 1. What the logs hold

| Set | Model | Each copy holds | Swarm sizes | Games per size |
|---|---|---|---|---|
| Stage 1, main grid | Qwen3-8B | 2 of 4 | 2, 4, 8, 16, 32 | 100 |
| Stage 1, 1-of-4 grid | Qwen3-8B | 1 of 4 | 4, 8, 16, 32 | 100 |
| Stage 1, random halves | Qwen3-8B | random 2 of 4 | 2, 4, 8, 16, 32 | 100 |
| Stage 1, 8 questions | Qwen3-8B | 4 of 8 | 2, 4, 8, 16, 32 | 99, 100 |
| Stage 1, Gemma | Gemma 3 27B | 2 of 4 | 2, 4, 8 | 100 |
| Stage 1, mixed swarms | 2 Qwen3-8B + 2 Gemma 3 27B | 2 of 4 | 4 | 100 |
| Stage 1, mixed swarms | 2 Qwen3-8B + 6 Gemma 3 27B | 2 of 4 | 8 | 100 |
| Stage 1, mixed swarms | 4 Qwen3-8B + 4 Gemma 3 27B | 2 of 4 | 8 | 100 |
| Stage 1, mixed swarms | 6 Qwen3-8B + 2 Gemma 3 27B | 2 of 4 | 8 | 100 |
| Our 4-round runs | Qwen3-4B | 2 of 4 | 2, 4, 8, 16, 32 | 100 |
| Our 4-round runs | Qwen3-8B | 2 of 4 | 4, 16, 64 | 100 |
| Our 4-round runs | Qwen3-14B | 2 of 4 | 2, 4, 8, 16, 32 | 100 |
| Our 4-round runs | Qwen3-32B | 2 of 4 | 2, 4, 8, 16, 32 | 100 |

## 2. Titles in round 1, by how many answers a copy holds

Round 1 is before any copy has seen another's page. Counted on N ≤ 8. Bare state name: the seeker read the page titled with the state alone, where holders almost never post. Title match: Σ_t r_t w_t, the chance that one seeker and one posting holder meet.

| Set | Model | Each copy holds | Seekers' most-read title | Its share | Seekers reading the bare state name | Holders' top title, share | Title match |
|---|---|---|---|---|---|---|---|
| Stage 1, main grid | Qwen3-8B | 2 of 4 | <S> construction occupation share | 51.4% | 12.3% | 44.4% | 26.2% |
| Stage 1, 1-of-4 grid | Qwen3-8B | 1 of 4 | <S> construction occupation share | 62.7% | 1.7% | 56.4% | 38.1% |
| Stage 1, random halves | Qwen3-8B | random 2 of 4 | <S> construction occupation share | 50.0% | 14.2% | 41.6% | 24.6% |
| Stage 1, 8 questions | Qwen3-8B | 4 of 8 | <S> construction occupation share | 46.9% | 30.4% | 53.6% | 27.1% |
| Stage 1, Gemma | Gemma 3 27B | 2 of 4 | <S> Construction Employment Share | 91.4% | 0.0% | 61.4% | 59.4% |
| Our 4-round runs | Qwen3-4B | 2 of 4 | <S> | 50.1% | 50.1% | 56.9% | 30.4% |
| Our 4-round runs | Qwen3-8B | 2 of 4 | <S> construction occupation share | 52.5% | 11.5% | 44.6% | 26.8% |
| Our 4-round runs | Qwen3-14B | 2 of 4 | <S> construction employment share | 44.7% | 0.1% | 29.0% | 14.9% |
| Our 4-round runs | Qwen3-32B | 2 of 4 | <S> construction employment share | 54.0% | 5.9% | 100.0% | 14.3% |

## 3. Find against the coverage term

Share of chances in which the seeker read a page holding the value. Coverage term: titles from N ≤ 8 of the same set, model and stage (round 1, or rounds 2 and later pooled), applied to each chance's K. Cells above N = 8 are held out. Observed: 95% interval from resampling games.

| Set | Model | Stage | N | Mean K | Observed find | Coverage term | Held out |
|---|---|---|---|---|---|---|---|
| Stage 1, main grid | Qwen3-8B | round 1 | 2 | 0.9 | 27.0 [19.0, 37.0] | 24.1 |  |
| Stage 1, main grid | Qwen3-8B | round 1 | 4 | 1.9 | 43.0 [35.0, 51.0] | 39.3 |  |
| Stage 1, main grid | Qwen3-8B | round 1 | 8 | 3.8 | 56.0 [50.2, 61.8] | 55.6 |  |
| Stage 1, main grid | Qwen3-8B | round 1 | 16 | 7.4 | 63.5 [58.9, 67.9] | 67.4 | yes |
| Stage 1, main grid | Qwen3-8B | round 1 | 32 | 15.0 | 72.7 [68.6, 76.4] | 75.6 | yes |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 2 | 0.9 | 19.3 [12.7, 26.7] | 21.5 |  |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 4 | 1.7 | 35.3 [28.5, 42.5] | 34.4 |  |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 8 | 3.4 | 48.4 [43.3, 53.8] | 49.5 |  |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 16 | 6.9 | 57.5 [53.8, 61.5] | 63.2 | yes |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 32 | 13.8 | 70.8 [67.2, 74.1] | 73.5 | yes |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 4 | 0.9 | 36.3 [28.7, 43.7] | 35.9 |  |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 8 | 1.9 | 56.2 [50.7, 61.5] | 53.4 |  |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 16 | 3.8 | 68.8 [64.9, 72.4] | 68.7 | yes |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 32 | 7.5 | 80.2 [77.1, 82.8] | 78.5 | yes |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 4 | 0.9 | 38.0 [32.2, 44.1] | 36.6 |  |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 8 | 1.8 | 54.3 [49.4, 59.3] | 53.7 |  |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 16 | 3.5 | 68.1 [65.6, 70.7] | 68.3 | yes |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 32 | 7.0 | 77.8 [75.9, 79.4] | 77.6 | yes |
| Stage 1, random halves | Qwen3-8B | round 1 | 2 | 0.9 | 23.5 [11.8, 35.3] | 23.2 |  |
| Stage 1, random halves | Qwen3-8B | round 1 | 4 | 1.6 | 29.0 [21.2, 38.1] | 32.5 |  |
| Stage 1, random halves | Qwen3-8B | round 1 | 8 | 3.3 | 45.9 [39.7, 52.0] | 49.4 |  |
| Stage 1, random halves | Qwen3-8B | round 1 | 16 | 7.1 | 65.4 [61.3, 69.4] | 64.3 | yes |
| Stage 1, random halves | Qwen3-8B | round 1 | 32 | 14.5 | 72.0 [68.0, 75.5] | 72.9 | yes |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 2 | 0.9 | 22.4 [13.5, 31.0] | 21.5 |  |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 4 | 1.5 | 31.3 [25.1, 37.7] | 30.8 |  |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 8 | 3.0 | 45.3 [40.9, 49.5] | 45.6 |  |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 16 | 6.3 | 66.5 [63.9, 69.0] | 61.0 | yes |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 32 | 13.2 | 79.6 [77.7, 81.4] | 74.1 | yes |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 2 | 0.9 | 28.0 [20.0, 37.0] | 25.2 |  |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 4 | 1.8 | 36.0 [29.0, 43.0] | 37.4 |  |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 8 | 3.7 | 42.5 [36.7, 48.0] | 49.0 |  |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 16 | 7.5 | 54.6 [49.2, 59.6] | 56.4 | yes |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 32 | 14.9 | 61.2 [56.1, 66.0] | 60.8 | yes |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 2 | 0.9 | 26.1 [18.1, 34.3] | 26.9 |  |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 4 | 1.9 | 40.4 [34.0, 47.2] | 43.5 |  |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 8 | 3.7 | 49.3 [43.6, 54.4] | 58.4 |  |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 16 | 7.2 | 55.9 [51.7, 60.4] | 69.3 | yes |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 32 | 14.4 | 64.3 [60.3, 68.4] | 79.8 | yes |
| Stage 1, Gemma | Gemma 3 27B | round 1 | 2 | 1.0 | 62.0 [53.0, 71.0] | 58.8 |  |
| Stage 1, Gemma | Gemma 3 27B | round 1 | 4 | 2.0 | 86.0 [80.0, 92.0] | 82.9 |  |
| Stage 1, Gemma | Gemma 3 27B | round 1 | 8 | 4.0 | 95.0 [90.8, 98.2] | 96.7 |  |
| Stage 1, Gemma | Gemma 3 27B | rounds 2+ | 2 | 0.4 | 24.0 [19.3, 29.0] | 23.2 |  |
| Stage 1, Gemma | Gemma 3 27B | rounds 2+ | 4 | 0.9 | 41.3 [36.0, 47.2] | 41.5 |  |
| Stage 1, Gemma | Gemma 3 27B | rounds 2+ | 8 | 1.9 | 52.1 [47.2, 57.2] | 55.7 |  |
| Our 4-round runs | Qwen3-4B | round 1 | 2 | 0.9 | 31.0 [22.0, 40.0] | 28.6 |  |
| Our 4-round runs | Qwen3-4B | round 1 | 4 | 1.9 | 44.0 [36.5, 51.5] | 45.0 |  |
| Our 4-round runs | Qwen3-4B | round 1 | 8 | 3.8 | 61.8 [55.2, 68.8] | 62.8 |  |
| Our 4-round runs | Qwen3-4B | round 1 | 16 | 7.7 | 74.2 [68.2, 79.4] | 77.3 | yes |
| Our 4-round runs | Qwen3-4B | round 1 | 32 | 15.5 | 76.6 [71.6, 80.9] | 85.9 | yes |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 2 | 0.9 | 29.0 [20.7, 38.0] | 30.9 |  |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 4 | 1.9 | 45.5 [37.8, 53.3] | 50.7 |  |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 8 | 3.8 | 60.2 [54.0, 65.7] | 69.9 |  |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 16 | 7.5 | 68.8 [64.4, 72.5] | 80.5 | yes |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 32 | 15.2 | 74.7 [71.1, 78.3] | 85.1 | yes |
| Our 4-round runs | Qwen3-8B | round 1 | 4 | 1.9 | 42.5 [35.0, 51.5] | 40.1 |  |
| Our 4-round runs | Qwen3-8B | round 1 | 16 | 7.4 | 65.0 [60.5, 69.6] | 68.7 | yes |
| Our 4-round runs | Qwen3-8B | round 1 | 64 | 30.2 | 78.8 [74.2, 82.7] | 83.0 | yes |
| Our 4-round runs | Qwen3-8B | rounds 2+ | 4 | 1.7 | 36.0 [28.5, 43.0] | 35.0 |  |
| Our 4-round runs | Qwen3-8B | rounds 2+ | 16 | 7.0 | 60.9 [56.9, 64.6] | 63.2 | yes |
| Our 4-round runs | Qwen3-8B | rounds 2+ | 64 | 27.6 | 79.8 [76.7, 82.3] | 81.4 | yes |
| Our 4-round runs | Qwen3-14B | round 1 | 2 | 0.5 | 7.0 [2.0, 12.0] | 7.2 |  |
| Our 4-round runs | Qwen3-14B | round 1 | 4 | 1.0 | 15.0 [10.0, 20.5] | 14.5 |  |
| Our 4-round runs | Qwen3-14B | round 1 | 8 | 2.1 | 25.2 [19.0, 31.8] | 25.8 |  |
| Our 4-round runs | Qwen3-14B | round 1 | 16 | 4.2 | 37.6 [31.6, 44.0] | 40.8 | yes |
| Our 4-round runs | Qwen3-14B | round 1 | 32 | 8.5 | 51.7 [46.2, 57.5] | 57.3 | yes |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 2 | 0.4 | 7.3 [3.0, 12.3] | 8.1 |  |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 4 | 0.9 | 10.5 [7.2, 14.2] | 15.2 |  |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 8 | 1.8 | 23.7 [19.4, 27.7] | 26.7 |  |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 16 | 3.1 | 30.5 [26.6, 34.2] | 38.5 | yes |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 32 | 6.1 | 44.7 [40.5, 49.0] | 54.9 | yes |
| Our 4-round runs | Qwen3-32B | round 1 | 2 | 0.0 | 0.0 [0.0, 0.0] | 0.0 |  |
| Our 4-round runs | Qwen3-32B | round 1 | 4 | 0.0 | 0.0 [0.0, 0.0] | 0.3 |  |
| Our 4-round runs | Qwen3-32B | round 1 | 8 | 0.0 | 0.8 [0.0, 2.0] | 0.6 |  |
| Our 4-round runs | Qwen3-32B | round 1 | 16 | 0.1 | 0.8 [0.0, 1.9] | 0.9 | yes |
| Our 4-round runs | Qwen3-32B | round 1 | 32 | 0.2 | 0.8 [0.2, 1.4] | 1.6 | yes |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 2 | 0.1 | 0.0 [0.0, 0.0] | 0.7 |  |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 4 | 0.2 | 2.0 [0.3, 4.0] | 1.9 |  |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 8 | 0.2 | 1.1 [0.2, 2.2] | 2.1 |  |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 16 | 0.3 | 4.3 [2.5, 6.7] | 3.8 | yes |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 32 | 0.8 | 12.1 [8.6, 15.6] | 8.2 | yes |

| Held-out cells | Cells | Median error | Largest error | Mean signed error | Inside the observed interval |
|---|---|---|---|---|---|
| Qwen3-8B, round 1 (stage 1 and ours) | 10 | 1.7 | 4.2 | +1.4 | 9 of 10 |
| Qwen3-8B, rounds 2+ | 10 | 4.1 | 15.5 | +3.0 | 5 of 10 |
| Qwen3-4B, 14B, 32B (ours), round 1 | 6 | 3.1 | 9.4 | +3.7 | 4 of 6 |
| Qwen3-4B, 14B, 32B (ours), rounds 2+ | 6 | 9.0 | 11.7 | +6.0 | 1 of 6 |

Errors in points (coverage term minus observed).

## 4. Equal holders: the stage 1 gap

Stage 1 compared swarms with the same number of holders per answer: each copy holding 2 of 4 answers, or 1 of 4 in a swarm twice the size. Round 1 find, with the coverage term given each grid's own titles and the other grid's (all from N ≤ 8).

| Holders per answer | Each copy holds | N | CR-v2, all rounds | Find, round 1 | Coverage, own titles | Coverage, the other grid's titles |
|---|---|---|---|---|---|---|
| 2 | 2 of 4 | 4 | 36.9 | 43.0 | 39.3 | 53.0 |
| 2 | 1 of 4 | 8 | 51.9 | 56.2 | 53.4 | 39.5 |
| 4 | 2 of 4 | 8 | 49.2 | 56.0 | 55.6 | 68.3 |
| 4 | 1 of 4 | 16 | 66.4 | 68.8 | 68.7 | 55.9 |
| 8 | 2 of 4 | 16 | 58.4 | 63.5 | 67.4 | 78.2 |
| 8 | 1 of 4 | 32 | 76.2 | 80.2 | 78.5 | 67.8 |

Which side of the titles carries the gap, in the 1-of-4 grid (coverage term, round 1):

| N | Observed | Reads and writes of the 1-of-4 grid | Its reads, main-grid writes | Main-grid reads, its writes | Reads and writes of the main grid |
|---|---|---|---|---|---|
| 8 | 56.2 | 53.4 | 44.7 | 44.0 | 39.5 |
| 16 | 68.8 | 68.7 | 62.1 | 56.9 | 55.9 |
| 32 | 80.2 | 78.5 | 73.8 | 65.5 | 67.8 |

## 5. Mixed swarms

Qwen3-8B and Gemma 3 27B on one wiki. Each model's seekers, with the coverage term built from the pure swarms' titles (Qwen from the main grid, Gemma from its own runs; N ≤ 8): 1 − Σ_t r_t Π_model (1 − w_t)^K_model. The titles come from the pure swarms only. K_model, the number of posters of each model, is counted in each mixed game; section 6 forecasts it from the pure swarms too.

| Swarm | Seekers | Stage | Observed find | Coverage term from the pure swarms | Chances |
|---|---|---|---|---|---|
| 2 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | round 1 | 30.0 [22.0, 39.0] | 29.1 | 100 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | rounds 2+ | 22.0 [15.0, 29.0] | 23.4 | 300 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | round 1 | 63.0 [53.0, 72.0] | 61.4 | 100 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 25.7 [20.7, 31.3] | 28.8 | 300 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Qwen3-8B | round 1 | 36.0 [26.0, 46.0] | 32.3 | 100 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Qwen3-8B | rounds 2+ | 26.3 [18.7, 34.0] | 26.7 | 300 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Gemma 3 27B | round 1 | 88.3 [82.3, 94.3] | 93.1 | 300 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 51.4 [45.4, 57.7] | 55.1 | 900 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Qwen3-8B | round 1 | 48.5 [40.5, 57.0] | 45.2 | 200 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Qwen3-8B | rounds 2+ | 37.7 [30.8, 43.8] | 37.4 | 600 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Gemma 3 27B | round 1 | 78.0 [70.5, 85.0] | 84.7 | 200 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 40.5 [34.5, 46.7] | 45.9 | 600 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | round 1 | 52.7 [46.0, 60.0] | 52.6 | 300 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | rounds 2+ | 41.8 [36.0, 46.9] | 44.7 | 900 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | round 1 | 62.0 [53.0, 72.0] | 64.8 | 100 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 33.0 [26.7, 40.0] | 36.0 | 300 |

Round 1: median error 3.0 points, largest 6.7; inside the observed interval in 8 of 8 cells.

Rounds 2+: median error 3.0 points, largest 5.4; inside the observed interval in 8 of 8 cells.

## 6. With the number of posters forecast as well

Sections 3 to 5 give the coverage term each target game's own K. Here K is forecast too. Each holder of the answer posts with chance π, the share of holders that had posted in swarms of N ≤ 8 of the same set, model and stage, independently of the others. The mean of (1 − w_t)^K over K ~ Binomial(h, π) is (1 − π w_t)^h, so the forecast is 1 − Σ_t r_t Π_model (1 − π_model w_t)^h_model, with h the holders of the answer. Nothing from the target cells is used.

| Set | Model | Stage | N | π, small swarms | Posting in the cell | Observed find | Given K | K forecast |
|---|---|---|---|---|---|---|---|---|
| Stage 1, main grid | Qwen3-8B | round 1 | 16 | 93.5 | 92.9 | 63.5 [58.9, 67.9] | 67.4 | 67.8 |
| Stage 1, main grid | Qwen3-8B | round 1 | 32 | 93.5 | 93.5 | 72.7 [68.6, 76.4] | 75.6 | 75.8 |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 16 | 84.1 | 86.8 | 57.5 [53.8, 61.5] | 63.2 | 63.3 |
| Stage 1, main grid | Qwen3-8B | rounds 2+ | 32 | 84.1 | 86.1 | 70.8 [67.2, 74.1] | 73.5 | 73.7 |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 16 | 93.2 | 94.2 | 68.8 [64.9, 72.4] | 68.7 | 68.7 |
| Stage 1, 1-of-4 grid | Qwen3-8B | round 1 | 32 | 93.2 | 94.0 | 80.2 [77.1, 82.8] | 78.5 | 78.4 |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 16 | 88.3 | 86.8 | 68.1 [65.6, 70.7] | 68.3 | 69.2 |
| Stage 1, 1-of-4 grid | Qwen3-8B | rounds 2+ | 32 | 88.3 | 87.7 | 77.8 [75.9, 79.4] | 77.6 | 77.9 |
| Stage 1, random halves | Qwen3-8B | round 1 | 16 | 92.7 | 91.9 | 65.4 [61.3, 69.4] | 64.3 | 65.0 |
| Stage 1, random halves | Qwen3-8B | round 1 | 32 | 92.7 | 93.3 | 72.0 [68.0, 75.5] | 72.9 | 73.1 |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 16 | 86.4 | 85.3 | 66.5 [63.9, 69.0] | 61.0 | 61.6 |
| Stage 1, random halves | Qwen3-8B | rounds 2+ | 32 | 86.4 | 85.4 | 79.6 [77.7, 81.4] | 74.1 | 74.5 |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 16 | 92.2 | 93.4 | 54.6 [49.2, 59.6] | 56.4 | 56.5 |
| Stage 1, 8 questions | Qwen3-8B | round 1 | 32 | 92.2 | 93.4 | 61.2 [56.1, 66.0] | 60.8 | 60.9 |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 16 | 92.3 | 89.7 | 55.9 [51.7, 60.4] | 69.3 | 70.8 |
| Stage 1, 8 questions | Qwen3-8B | rounds 2+ | 32 | 92.3 | 90.1 | 64.3 [60.3, 68.4] | 79.8 | 80.8 |
| Our 4-round runs | Qwen3-4B | round 1 | 16 | 95.0 | 96.2 | 74.2 [68.2, 79.4] | 77.3 | 77.1 |
| Our 4-round runs | Qwen3-4B | round 1 | 32 | 95.0 | 96.6 | 76.6 [71.6, 80.9] | 85.9 | 85.8 |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 16 | 95.1 | 94.0 | 68.8 [64.4, 72.5] | 80.5 | 80.9 |
| Our 4-round runs | Qwen3-4B | rounds 2+ | 32 | 95.1 | 95.1 | 74.7 [71.1, 78.3] | 85.1 | 85.1 |
| Our 4-round runs | Qwen3-8B | round 1 | 16 | 93.0 | 92.9 | 65.0 [60.5, 69.6] | 68.7 | 69.1 |
| Our 4-round runs | Qwen3-8B | round 1 | 64 | 93.0 | 94.3 | 78.8 [74.2, 82.7] | 83.0 | 83.0 |
| Our 4-round runs | Qwen3-8B | rounds 2+ | 16 | 84.3 | 87.6 | 60.9 [56.9, 64.6] | 63.2 | 62.9 |
| Our 4-round runs | Qwen3-8B | rounds 2+ | 64 | 84.3 | 86.3 | 79.8 [76.7, 82.3] | 81.4 | 81.8 |
| Our 4-round runs | Qwen3-14B | round 1 | 16 | 52.8 | 53.1 | 37.6 [31.6, 44.0] | 40.8 | 44.0 |
| Our 4-round runs | Qwen3-14B | round 1 | 32 | 52.8 | 53.1 | 51.7 [46.2, 57.5] | 57.3 | 63.4 |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 16 | 43.8 | 38.8 | 30.5 [26.6, 34.2] | 38.5 | 46.2 |
| Our 4-round runs | Qwen3-14B | rounds 2+ | 32 | 43.8 | 38.1 | 44.7 [40.5, 49.0] | 54.9 | 67.0 |
| Our 4-round runs | Qwen3-32B | round 1 | 16 | 1.0 | 1.1 | 0.8 [0.0, 1.9] | 0.9 | 1.1 |
| Our 4-round runs | Qwen3-32B | round 1 | 32 | 1.0 | 1.2 | 0.8 [0.2, 1.4] | 1.6 | 2.0 |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 16 | 5.4 | 4.3 | 4.3 [2.5, 6.7] | 3.8 | 4.8 |
| Our 4-round runs | Qwen3-32B | rounds 2+ | 32 | 5.4 | 5.2 | 12.1 [8.6, 15.6] | 8.2 | 9.1 |

| Held-out cells | Cells | Given K: median, largest error | Inside | K forecast: median, largest error | Inside |
|---|---|---|---|---|---|
| Qwen3-8B, round 1 (stage 1 and ours) | 10 | 1.7, 4.2 | 9 of 10 | 1.8, 4.3 | 9 of 10 |
| Qwen3-8B, rounds 2+ | 10 | 4.1, 15.5 | 5 of 10 | 3.9, 16.5 | 5 of 10 |
| Qwen3-4B, 14B, 32B (ours), round 1 | 6 | 3.1, 9.4 | 4 of 6 | 4.6, 11.6 | 3 of 6 |
| Qwen3-4B, 14B, 32B (ours), rounds 2+ | 6 | 9.0, 11.7 | 1 of 6 | 11.2, 22.2 | 2 of 6 |
| Qwen3-14B (ours), round 1 | 2 | 4.4, 5.6 | 2 of 2 | 9.0, 11.6 | 1 of 2 |

Errors in points. For Qwen3-8B in round 1 the forecast of K costs nothing. For Qwen3-14B it does. The mean K is the same either way (the two posting columns), so the difference is in how the posters are spread: in the games they are bunched on some questions, and independent holders would spread them over more.

Mixed swarms, with each model's π from its pure swarms:

| Swarm | Seekers | Stage | Observed find | Given K | K forecast |
|---|---|---|---|---|---|
| 2 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | round 1 | 30.0 [22.0, 39.0] | 29.1 | 28.7 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | rounds 2+ | 22.0 [15.0, 29.0] | 23.4 | 23.2 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | round 1 | 63.0 [53.0, 72.0] | 61.4 | 61.1 |
| 2 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 25.7 [20.7, 31.3] | 28.8 | 31.5 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Qwen3-8B | round 1 | 36.0 [26.0, 46.0] | 32.3 | 32.2 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Qwen3-8B | rounds 2+ | 26.3 [18.7, 34.0] | 26.7 | 26.0 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Gemma 3 27B | round 1 | 88.3 [82.3, 94.3] | 93.1 | 93.1 |
| 2 Qwen3-8B + 6 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 51.4 [45.4, 57.7] | 55.1 | 62.7 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Qwen3-8B | round 1 | 48.5 [40.5, 57.0] | 45.2 | 45.5 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Qwen3-8B | rounds 2+ | 37.7 [30.8, 43.8] | 37.4 | 38.3 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Gemma 3 27B | round 1 | 78.0 [70.5, 85.0] | 84.7 | 84.6 |
| 4 Qwen3-8B + 4 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 40.5 [34.5, 46.7] | 45.9 | 52.6 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | round 1 | 52.7 [46.0, 60.0] | 52.6 | 52.9 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Qwen3-8B | rounds 2+ | 41.8 [36.0, 46.9] | 44.7 | 45.8 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | round 1 | 62.0 [53.0, 72.0] | 64.8 | 64.6 |
| 6 Qwen3-8B + 2 Gemma 3 27B | Gemma 3 27B | rounds 2+ | 33.0 [26.7, 40.0] | 36.0 | 38.2 |

Round 1: given K, median error 3.0 points, largest 6.7, inside the observed interval in 8 of 8 cells; K forecast, median 2.8, largest 6.6, inside in 8 of 8.

Rounds 2+: given K, median error 3.0 points, largest 5.4, inside the observed interval in 8 of 8 cells; K forecast, median 4.7, largest 12.1, inside in 5 of 8.
