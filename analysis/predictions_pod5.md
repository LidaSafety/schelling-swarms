# Predictions for the fifth pod session (pod H)

Written and committed before these runs start. Run names are those of `configs/experiments.yaml` in the benchmark (entries 17 to 19). Rerun with `python predict_pod5.py` (it will not overwrite this file). Every forecast is counted on the 4-round logs at N ≤ 8 with the model of `fuller.py` and the paper's second version (§4.1, Table 2), unchanged; nothing is counted on the runs it forecasts. Intervals: 95%, from resampling 200 times the whole games of the N ≤ 8 swarms each forecast is counted on. Percent throughout.

Setting, as the reference runs (`results/swarm4round/`): the rule in force, unaware, no page listing, no log, games 10000 onwards, so each game pairs with the reference game of the same quiz. These runs are on one H200; the logs the forecasts are counted on were made on other GPUs, and in `wikigame_hwcheck` only 133 of 320 copies repeated their run word for word across GPUs. Each model's N = 8 cell is therefore rerun first in this session as a control (reported, not registered).

## Run 17: the scaling forecast at N = 64 and 128

Five conditions: Qwen3-4B, 8B, 14B and 32B with 2 answers per copy (f_inv = 2), and Qwen3-8B with 1 answer (f_inv = 4). N = 64: 100 games; N = 128: 50 games. Qwen3-8B with 2 answers was run at N = 64 before (`swarm_ref_8b_n64`, 78.3 [75.3, 81.0]); its forecast there is in the paper's Table 18, so here it runs at N = 128 only. Versions: *fuller* is the registered forecast (posting and titles counted by round; use counted apart on pages that also show a guess). *Simple* (one count for round 1 and one for rounds 2–4, one share of use) and *line in log2 N* (a straight line through the condition's own N ≤ 8 rates) are the alternatives it is scored against. *Lead* (use with a guess shown depends on whether true posts outnumber guesses) was tried after the earlier runs and is reported beside the others, not as the forecast.

| Condition | N | **Fuller (registered)** | Simple | Line in log2 N | Lead | N = 8, earlier runs (the control's reference) |
|---|---|---|---|---|---|---|
| Qwen3-4B, 2 answers | 64 | **42.2** [37.7, 46.5] | 52.6 [49.7, 55.9] | 47.8 [33.6, 63.6] | 41.4 [37.2, 45.8] | 33.7 [29.2, 38.1] |
| Qwen3-4B, 2 answers | 128 | **43.1** [38.6, 47.5] | 54.6 [51.6, 57.9] | 52.6 [34.5, 73.3] | 41.9 [37.6, 46.7] |  |
| Qwen3-8B, 2 answers | 128 | **75.7** [70.6, 80.2] | 85.6 [83.7, 87.2] | 100.0 [84.7, 100.0] | 76.8 [72.5, 83.9] | 49.2 [44.1, 53.8] |
| Qwen3-8B, 1 answer | 64 | **68.3** [61.8, 75.0] | 80.9 [78.6, 82.9] | 98.2 [79.1, 100.0] | 77.2 [71.0, 82.8] | 51.9 [47.6, 56.0] |
| Qwen3-8B, 1 answer | 128 | **66.0** [58.4, 74.9] | 86.2 [83.7, 88.3] | 100.0 [88.5, 100.0] | 79.4 [70.3, 87.2] |  |
| Qwen3-14B, 2 answers | 64 | **31.4** [23.2, 38.8] | 60.7 [55.9, 66.2] | 32.1 [21.4, 42.1] | 45.6 [36.7, 51.0] | 17.0 [14.0, 20.4] |
| Qwen3-14B, 2 answers | 128 | **31.0** [21.1, 39.7] | 66.8 [61.4, 72.4] | 37.4 [23.8, 49.9] | 51.7 [39.4, 57.3] |  |
| Qwen3-32B, 2 answers | 64 | **10.4** [5.8, 14.6] | 10.9 [5.7, 17.1] | 2.0 [0.7, 3.8] | 10.4 [5.8, 14.6] | 0.6 [0.1, 1.4] |
| Qwen3-32B, 2 answers | 128 | **16.9** [9.9, 22.6] | 18.1 [9.3, 27.0] | 2.3 [0.7, 4.5] | 16.9 [9.9, 22.6] |  |

A line in log2 N above 100 is shown as 100.

Registered, each cell against its measured CR (95% interval from resampling whole games):

- **Accuracy.** The fuller forecast is within 8 points of the measured CR (8 is its largest error on the 16 held-out swarms of the paper's Table 2). Counted cell by cell; the run holds if at least 7 of the 9 cells do.
- **Against the simple model**, in the cells where the two differ by 5 points or more (Qwen3-4B, 2 answers at 64: 42.2 against 52.6; Qwen3-4B, 2 answers at 128: 43.1 against 54.6; Qwen3-8B, 2 answers at 128: 75.7 against 85.6; Qwen3-8B, 1 answer at 64: 68.3 against 80.9; Qwen3-8B, 1 answer at 128: 66.0 against 86.2; Qwen3-14B, 2 answers at 64: 31.4 against 60.7; Qwen3-14B, 2 answers at 128: 31.0 against 66.8): the measured CR is closer to the fuller forecast in a majority of them.
- **Against the line in log2 N**, in the cells where the two differ by 5 points or more (Qwen3-4B, 2 answers at 64: 42.2 against 47.8; Qwen3-4B, 2 answers at 128: 43.1 against 52.6; Qwen3-8B, 2 answers at 128: 75.7 against 100.0; Qwen3-8B, 1 answer at 64: 68.3 against 98.2; Qwen3-8B, 1 answer at 128: 66.0 against 100.0; Qwen3-14B, 2 answers at 128: 31.0 against 37.4; Qwen3-32B, 2 answers at 64: 10.4 against 2.0; Qwen3-32B, 2 answers at 128: 16.9 against 2.3): the measured CR is closer to the fuller forecast in a majority of them.
- **Any against lead at large N**, in the cells where the two forms of step 3 differ by 5 points or more (Qwen3-8B, 1 answer at 64: 68.3 against 77.2; Qwen3-8B, 1 answer at 128: 66.0 against 79.4; Qwen3-14B, 2 answers at 64: 31.4 against 45.6; Qwen3-14B, 2 answers at 128: 31.0 against 51.7): which the measured CR is closer to, cell by cell. Lead was tried after the earlier runs, so this is a test of it, not of the registered forecast.
- Readings fixed now: if the fuller model fails the accuracy condition, the paper's claim that a model counted on N ≤ 8 forecasts larger swarms is limited to N ≤ 64. If it passes but loses to the simple model or the line, the refinements are not what carries the forecast to N = 128.

Reported, not registered: the two halves of each cell (rounds 1 and 3, rounds 2 and 4) against the fuller model's forecast for each; and the N = 8 controls against the earlier N = 8 runs.

## Run 18: a guess planted before round 1 (N = 16)

Qwen3-4B, 8B and 14B, 2 answers per copy, 100 games, three arms on the same games: **none**, **wrong** (a value at least 1.0 from the truth that no question or note in the game has) and **true**. In each game the plant goes on one question of each half of the quiz (one of rounds 1 and 3, one of rounds 2 and 4, chosen by seed and the same in every arm), as one note, *"The share of employed workers in construction occupations in ⟨State⟩ is ⟨value⟩%."*, on each of the seven titles seekers read most at N ≤ 8 (`scb/swarm.py` PLANT_TITLES; together 87% of 4B's reads, 79% of 8B's and 81% of 14B's). Planted notes are never another copy's, so they never count as found, seen or used, and CR is unchanged in meaning.

**What it tests.** Step 3 of the model: a finder shown a guess beside the true value uses the true value less (u with a guess shown, against u with none). In the logs this is observational: pages with a guess differ in title, round and model from pages without one. Every copy writes before it reads, and a planted note sits on a question's own titles, so the plant can change only what a seeker does with the page it reads. It also decides whether the cost depends on how many copies post the true value. Under *any* (the paper's step 3) it does not; the logs suggest it does (in large Qwen3-8B swarms a finder shown a guess beside the true value uses it 82–95% of the time, against 57–67% at N ≤ 8). The *lead* form counted on N ≤ 8 cannot show this here: it differs from *any* by 3 points or less for every model, because few pages at N ≤ 8 carry several true posts. So the dose is registered directly, below.

Reached: a seeker of a planted question that reads one of its planted titles; in the none arm, the same titles. Reached finder: a reached seeker whose page also shows another copy's true value. Both groups are defined the same way in every arm, each from its own logs.

| Model | Reach | Reached finders, of reached | u, no guess shown (N ≤ 8) | u, guess shown (finders at N ≤ 8) | **Use by reached finders, none** | **wrong (any)** | **wrong − none (any)** | wrong − none (lead) |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 86.6 [84.9, 88.2] | 86.8 [84.1, 88.9] | 74.2 | 50.2 (916) | **51.9** [46.9, 56.5] | **50.2** [44.5, 55.1] | **-1.7** [-2.4, -1.2] | -4.2 [-5.7, -2.7] |
| Qwen3-8B | 79.3 [77.1, 81.4] | 73.3 [70.7, 75.8] | 99.5 | 67.4 (46) | **97.9** [96.6, 99.2] | **67.4** [47.2, 84.9] | **-30.5** [-50.3, -13.8] | -27.6 [-37.4, -3.6] |
| Qwen3-14B | 82.7 [80.9, 84.6] | 48.1 [44.2, 51.6] | 88.4 | 33.3 (129) | **62.9** [55.5, 68.6] | **33.3** [20.7, 44.7] | **-29.6** [-38.0, -23.3] | -27.0 [-34.3, -21.2] |

| Model | CR on planted questions, none | wrong (any) | **wrong − none (any)** | wrong − none (lead) | Reached seekers with no copy's true value: take the plant, wrong arm |
|---|---|---|---|---|---|
| Qwen3-4B | 39.4 [35.5, 43.4] | 38.1 [33.9, 42.1] | **-1.3** [-1.7, -0.9] | -3.1 [-4.3, -2.1] | 61.8 |
| Qwen3-8B | 60.6 [58.3, 62.9] | 42.9 [31.8, 52.6] | **-17.8** [-29.6, -8.0] | -16.0 [-22.2, -2.1] | 90.1 |
| Qwen3-14B | 26.3 [23.2, 29.3] | 14.6 [9.2, 18.7] | **-11.8** [-15.1, -8.9] | -10.7 [-13.7, -8.0] | 74.6 |

Registered, against the none arm of this session (intervals from resampling games, paired by game):

- **Qwen3-4B, reached finders:** use of the true value changes by -1.7 points [-2.4, -1.2] in the wrong arm (*any*). Counted as holding if the observed change is within 10 points of it.
- **Qwen3-8B, reached finders:** use of the true value changes by -30.5 points [-50.3, -13.8] in the wrong arm (*any*). Counted as holding if the observed change is within 10 points of it.
- **Qwen3-14B, reached finders:** use of the true value changes by -29.6 points [-38.0, -23.3] in the wrong arm (*any*). Counted as holding if the observed change is within 10 points of it.
- **The dose.** In the wrong arm, reached finders whose page shows the true value from 3 or more copies use it more often than those whose page shows it from 1 copy, by 10 points or more (Qwen3-8B and Qwen3-14B pooled; each reported alone, and the none arm beside it). Readings fixed now: a rise of 10 points or more → the cost of a guess falls as more copies post the true value, step 3 should count posts and not only whether a guess is there, and this is the likely cause of the fuller model's large-N miss for Qwen3-8B; a rise under 10 points → one guess is enough, and that miss has another cause.
- **The planted value is taken.** In the wrong arm, reached seekers whose page shows no copy's true value give the planted value at the rate of a lone wrong number in the logs (last column; `roles.md` 11.3, all N), within 15 points either way.
- **No change where the plant cannot act.** Holders who post, by round, and CR on the unplanted questions: within 5 points of the none arm in both plant arms.
- **The true arm.** Under *any*, an extra true note is not a guess, so reached finders' use is within 5 points of the none arm. Reported with the share of reached seekers with no copy's true value that give the planted true value (they cannot count in CR).

Reported, not registered: what holders of the planted question do in their later rounds (a holder reads in every round, so a copy can meet a plant on another question before its own holder round), and whether copies post the planted value onward.

## Run 19 (optional): 4 answers per copy, first stage

Q = 8 questions with f_inv = 2, so each copy holds 4 answers and h = N/2, as in the main runs; only the answers per copy change. Qwen3-4B, 14B and 32B at N = 2, 4 and 8, 100 games. Qwen3-8B already has this setting at N = 2–32 from the hackathon (`4-round-results/swarm_q8_n*`). Nothing is registered for this stage: it gives the counts a forecast for N = 16 and 32 needs, which will be committed before any second stage runs.
