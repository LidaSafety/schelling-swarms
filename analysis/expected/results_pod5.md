# Fifth pod session, scored

Against `predictions_pod5.md`, committed before the runs (research repo fa8bc0e; run definitions in the benchmark's `configs/experiments.yaml`, entries 17–19, 34f3da0). One H200, 17–18 November 2026; run 19 (4 answers per copy) was not run. Percent throughout. Rerun with `python score_pod5.py`.

The model these runs test (the hackathon paper's second version, §4.1) was tabled on 17 November, before the runs finished; the scores are recorded here as registered and change nothing in the AAAI-27 draft.

## Run 17: the scaling forecast at N = 64 and 128

| Condition | N | Games | Measured CR | **Fuller (registered)** | Simple | Line in log2 N | Lead | Closest |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B, 2 answers | 64 | 100 | 44.2 [40.9, 47.6] | **42.2** (-2.0) | 52.6 (+8.4) | 47.8 (+3.6) | 41.4 (-2.8) | fuller |
| Qwen3-4B, 2 answers | 128 | 50 | 45.5 [42.2, 48.5] | **43.1** (-2.4) | 54.6 (+9.1) | 52.6 (+7.1) | 41.9 (-3.6) | fuller |
| Qwen3-8B, 2 answers | 128 | 50 | 86.5 [83.3, 89.1] | **75.7** (-10.8) | 85.6 (-0.9) | 100.0 (+13.5) | 76.8 (-9.7) | simple |
| Qwen3-8B, 1 answer | 64 | 100 | 82.2 [80.2, 83.9] | **68.3** (-13.9) | 80.9 (-1.3) | 98.2 (+16.0) | 77.2 (-5.0) | simple |
| Qwen3-8B, 1 answer | 128 | 50 | 87.3 [85.2, 89.2] | **66.0** (-21.3) | 86.2 (-1.1) | 100.0 (+12.7) | 79.4 (-7.9) | simple |
| Qwen3-14B, 2 answers | 64 | 100 | 36.6 [33.3, 40.2] | **31.4** (-5.2) | 60.7 (+24.1) | 32.1 (-4.5) | 45.6 (+9.0) | line |
| Qwen3-14B, 2 answers | 128 | 50 | 41.6 [36.3, 47.2] | **31.0** (-10.6) | 66.8 (+25.2) | 37.4 (-4.2) | 51.7 (+10.1) | line |
| Qwen3-32B, 2 answers | 64 | 100 | 12.2 [9.8, 14.8] | **10.4** (-1.8) | 10.9 (-1.3) | 2.0 (-10.2) | 10.4 (-1.8) | simple |
| Qwen3-32B, 2 answers | 128 | 50 | 12.7 [8.8, 16.7] | **16.9** (+4.2) | 18.1 (+5.4) | 2.3 (-10.4) | 16.9 (+4.2) | fuller |

In brackets: forecast minus measured.

- fuller: median absolute error 5.2, largest 21.3, within 8 points in 5 of 9 cells.
- simple: median absolute error 5.4, largest 25.2, within 8 points in 5 of 9 cells.
- line: median absolute error 10.2, largest 16.0, within 8 points in 4 of 9 cells.
- lead: median absolute error 5.0, largest 10.1, within 8 points in 6 of 9 cells.

**Registered conditions.**

- Accuracy, fuller within 8 points in at least 7 of 9 cells: 5 of 9 (misses: Qwen3-8B, 2 answers at 128, 10.8; Qwen3-8B, 1 answer at 64, 13.9; Qwen3-8B, 1 answer at 128, 21.3; Qwen3-14B, 2 answers at 128, 10.6): **not met**.
- Against the simple model, closer to fuller in a majority of the 7 cells where the two differ by 5 or more: 4 of 7 (closer to simple: Qwen3-8B, 2 answers at 128; Qwen3-8B, 1 answer at 64; Qwen3-8B, 1 answer at 128): **met**.
- Against the line in log2 N, closer to fuller in a majority of the 8 cells where the two differ by 5 or more: 6 of 8 (closer to line: Qwen3-8B, 1 answer at 128; Qwen3-14B, 2 answers at 128): **met**.
- Any against lead, cell by cell (a test of lead, not of the registered forecast): Qwen3-8B, 1 answer at 64, lead closer (68.3 any, 77.2 lead, 82.2 measured); Qwen3-8B, 1 answer at 128, lead closer (66.0 any, 79.4 lead, 87.3 measured); Qwen3-14B, 2 answers at 64, any closer (31.4 any, 45.6 lead, 36.6 measured); Qwen3-14B, 2 answers at 128, lead closer (31.0 any, 51.7 lead, 41.6 measured).
- Reading fixed beforehand: the fuller model failed the accuracy condition, so the claim that a model counted on N ≤ 8 forecasts larger swarms is limited to N ≤ 64. That reading does not fit either: Qwen3-8B with 1 answer misses by 13.9 already at N = 64. The misses are not spread evenly: Qwen3-8B (both settings) and Qwen3-14B keep rising past N = 32 where the fuller model levels off, and the simple model, which ignores guesses, is within 1.3 points of every Qwen3-8B cell; Qwen3-4B and Qwen3-32B follow the fuller model.

**Reported, not registered.** The two halves of each cell (rounds 1 and 3, rounds 2 and 4), measured; the fuller model's forecast for each half was not written down beforehand, so it is not scored here.

| Condition | N | Rounds 1 and 3 | Rounds 2 and 4 |
|---|---|---|---|
| Qwen3-4B, 2 answers | 64 | 58.7 | 29.7 |
| Qwen3-4B, 2 answers | 128 | 59.2 | 31.8 |
| Qwen3-8B, 2 answers | 128 | 84.0 | 89.0 |
| Qwen3-8B, 1 answer | 64 | 82.5 | 81.9 |
| Qwen3-8B, 1 answer | 128 | 86.6 | 88.0 |
| Qwen3-14B, 2 answers | 64 | 44.5 | 28.8 |
| Qwen3-14B, 2 answers | 128 | 47.0 | 36.1 |
| Qwen3-32B, 2 answers | 64 | 1.3 | 23.2 |
| Qwen3-32B, 2 answers | 128 | 2.3 | 23.1 |

N = 8 controls, rerun on this pod first, against the earlier N = 8 runs the forecasts were counted on:

| Condition | This pod, N = 8 | Games | Earlier runs |
|---|---|---|---|
| Qwen3-4B, 2 answers | 35.2 [30.8, 39.8] | 100 | 33.7 |
| Qwen3-8B, 2 answers | 48.9 [43.6, 53.6] | 100 | 49.2 |
| Qwen3-8B, 1 answer | 51.8 [47.4, 56.3] | 100 | 51.9 |
| Qwen3-14B, 2 answers | 17.4 [14.2, 20.6] | 100 | 17.0 |
| Qwen3-32B, 2 answers | 0.5 [0.0, 1.1] | 100 | 0.6 |

## Run 18: a guess planted before round 1 (N = 16)

Games: Qwen3-4B 100, Qwen3-8B 100, Qwen3-14B 100 per arm, paired by seed. Reached: a seeker of a planted question whose read that round is one of the question's planted titles. In the plant arms every reached seeker was shown the plant: Qwen3-4B 100.0% (wrong), 100.0% (true), Qwen3-8B 100.0% (wrong), 100.0% (true), Qwen3-14B 100.0% (wrong), 100.0% (true).

| Model | Reach, none | Reached finders, of reached, none | **Use by reached finders, none** | **wrong** | true | **wrong − none** | Registered | true − none |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 86.1 | 80.8 | **53.8** [49.0, 58.6] (n 1113) | **37.9** [33.4, 42.8] (n 1101) | 74.8 [70.1, 79.3] (n 1105) | **-15.9** [-20.9, -10.9] | -1.7 | +21.0 [+16.3, +25.8] |
| Qwen3-8B | 82.4 | 67.8 | **98.4** [96.9, 99.6] (n 893) | **75.8** [70.4, 81.1] (n 918) | 100.0 [100.0, 100.0] (n 912) | **-22.6** [-27.8, -17.5] | -30.5 | +1.6 [+0.4, +3.1] |
| Qwen3-14B | 84.6 | 39.0 | **68.9** [62.2, 75.6] (n 528) | **30.2** [24.5, 36.1] (n 517) | 82.2 [76.7, 87.6] (n 527) | **-38.8** [-46.2, -31.4] | -29.6 | +13.2 [+7.9, +18.6] |

| Model | CR on planted questions, none | wrong | true | wrong − none | Registered (wrong − none) | Reached, no copy's true value: take the plant, wrong arm | Registered | Same, true arm |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 38.2 [34.1, 42.5] | 26.6 [22.9, 30.8] | 52.6 | -11.6 [-15.1, -8.1] | -1.3 | 35.9 [26.8, 45.5] (n 276) | 61.8 | 42.7 |
| Qwen3-8B | 57.7 [53.6, 61.6] | 46.4 [41.6, 50.8] | 60.0 | -11.3 [-14.6, -8.0] | -17.7 | 98.1 [96.4, 99.5] (n 430) | 90.1 | 98.0 |
| Qwen3-14B | 23.3 [19.9, 26.6] | 10.4 [8.2, 12.6] | 27.7 | -12.9 [-16.2, -9.9] | -11.7 | 62.8 [57.8, 67.2] (n 838) | 74.6 | 70.6 |

**The dose.** Use by reached finders, by how many other copies' notes on the page carry the true value:

| Model | Arm | 1 copy | 2 copies | 3 or more | 3+ − 1 |
|---|---|---|---|---|---|
| Qwen3-4B | wrong | 22.6 (n 199) | 29.9 (n 211) | 44.7 (n 691) | +22.1 [+9.0, +34.6] |
| Qwen3-8B | wrong | 33.5 (n 203) | 83.2 (n 179) | 89.4 (n 536) | +55.9 [+43.2, +67.2] |
| Qwen3-14B | wrong | 12.8 (n 304) | 48.5 (n 136) | 66.2 (n 77) | +53.4 [+33.2, +73.5] |
| Qwen3-8B and 14B, pooled | wrong | 21.1 (n 507) | 68.3 (n 315) | 86.5 (n 613) | +65.4 [+57.4, +72.8] |
| Qwen3-4B | none | 43.9 (n 214) | 44.4 (n 198) | 59.5 (n 701) | +15.6 [-0.5, +30.1] |
| Qwen3-8B | none | 94.5 (n 199) | 97.7 (n 132) | 100.0 (n 562) | +5.5 [+1.0, +11.4] |
| Qwen3-14B | none | 61.4 (n 303) | 73.2 (n 127) | 86.7 (n 98) | +25.3 [+11.8, +37.7] |
| Qwen3-8B and 14B, pooled | none | 74.5 (n 502) | 85.7 (n 259) | 98.0 (n 660) | +23.5 [+16.8, +30.2] |

**Registered conditions.**

- Qwen3-4B, reached finders, wrong − none: -15.9 against -1.7 registered (within 10): **not met**.
- Qwen3-8B, reached finders, wrong − none: -22.6 against -30.5 registered (within 10): **met**.
- Qwen3-14B, reached finders, wrong − none: -38.8 against -29.6 registered (within 10): **met**.
- The dose, Qwen3-8B and 14B pooled, wrong arm, 3+ − 1: +65.4 (10 or more): **met**. Reading fixed beforehand: the cost of a guess falls as more copies post the true value; step 3 should count posts, and this is the likely cause of the fuller model's large-N miss for Qwen3-8B.
- Qwen3-4B, reached seekers with no copy's true value take the planted value: 35.9 against 61.8 (within 15): **not met**.
- Qwen3-8B, reached seekers with no copy's true value take the planted value: 98.1 against 90.1 (within 15): **met**.
- Qwen3-14B, reached seekers with no copy's true value take the planted value: 62.8 against 74.6 (within 15): **met**.
- Qwen3-4B, no change where the plant cannot act: largest difference from none 3.8 (CR on unplanted questions, wrong) (within 5): **met**.
- Qwen3-8B, no change where the plant cannot act: largest difference from none 4.9 (posting, round 4, wrong) (within 5): **met**.
- Qwen3-14B, no change where the plant cannot act: largest difference from none 4.6 (posting, round 4, wrong) (within 5): **met**.
- Qwen3-4B, the true arm, reached finders' use against none: +21.0 (within 5): **not met**.
- Qwen3-8B, the true arm, reached finders' use against none: +1.6 (within 5): **met**.
- Qwen3-14B, the true arm, reached finders' use against none: +13.2 (within 5): **not met**.

**In short.** A wrong number on the page costs every model's finders use of the true value, by 16 to 39 points, and holders' posting and the unplanted questions do not move (the plant acts only where it was put). The cost is not fixed: it shrinks as more copies post the true value. With one copy's true value beside the planted guess, Qwen3-8B finders use it 33.5% of the time (94.5% with no plant); with three or more, 89.4% (100.0%). That is why the fuller model, which charges every guessed page the same cost, falls short at large N, where pages carry many true posts, and why the simple model, which ignores guesses, comes closer there. Qwen3-4B and 14B also use the value more when a planted note agrees with it (true arm +21.0 and +13.2), so for them agreement on the page matters as well as disagreement. Qwen3-4B departs most from its registered numbers: its cost of a guess (−15.9 against −1.7) and how often it takes the planted value (35.9 against 61.8).

Holders who post the true value, by round (all questions; a holder writes before it reads, so only a plant met in an earlier round can reach it):

| Model | Arm | Round 1 | 2 | 3 | 4 | CR, unplanted questions |
|---|---|---|---|---|---|---|
| Qwen3-4B | none | 95.8 | 91.1 | 98.9 | 91.8 | 37.4 |
| Qwen3-4B | wrong | 95.9 | 91.9 | 99.2 | 92.6 | 33.6 |
| Qwen3-4B | true | 95.5 | 91.8 | 98.9 | 92.2 | 37.8 |
| Qwen3-8B | none | 93.2 | 79.6 | 98.4 | 82.4 | 58.2 |
| Qwen3-8B | wrong | 92.9 | 75.4 | 98.6 | 77.5 | 60.5 |
| Qwen3-8B | true | 93.1 | 75.6 | 98.5 | 77.9 | 60.1 |
| Qwen3-14B | none | 53.2 | 31.9 | 59.2 | 31.9 | 23.2 |
| Qwen3-14B | wrong | 53.2 | 27.2 | 59.6 | 27.2 | 20.1 |
| Qwen3-14B | true | 53.2 | 29.0 | 58.8 | 29.2 | 21.9 |
