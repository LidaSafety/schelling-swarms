# Forecasts for the planted notes (a test of precedent)

Written before these runs start, and committed before the first run. The paper's evidence that late holders post more
after reading longer pages is observational: longer pages lie under the titles most copies choose. These runs set the
number of notes on one title by design.

Runs: pairs of Gemma 3 27B and of Qwen3-8B on the fixed prompt (unaware, rule in force, no page list, `--log none`),
4 questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call, Qwen3 with thinking off, the models
served as for their base pairs. Before round 1, k notes sit on the model's favourite round-1 title for question 1,
"⟨State⟩ Construction Employment Share" for Gemma and "⟨State⟩ construction occupation share" for Qwen3-8B, in the
formats that model's own copies write there, weighted as counted on round-1 posts under that title in its 8-copy
swarms (`scb/swarm.py` PLANT_STYLES, `patches/plant-notes-runner.patch`). Planted notes are shown like any note but
never count as another copy's. Seeds from 10000 in every arm, so the arms play the same quizzes
(`scripts/run_plant_pairs.sh`).

| Arm | Planted notes | They carry | Gemma games | Qwen3-8B games |
|---|---|---|---|---|
| 0 | 0 | | 200 | 400 |
| 4 | 4 | question 1's true value, as its holders write | 200 | 400 |
| 16 | 16 | question 1's true value | 200 | 400 |
| 32 | 32 | question 1's true value | 200 | 400 |
| 16, no number | 16 | no number, as seekers write ("Unknown. Need to find data for ⟨State⟩.") | 200 | 400 |

Outcome: the late holders (the copy that first holds an answer in round 2; in a pair, copy A) whose round-1 read was the
planted title (in arm 0, the same title): the share that post their round-2 answer in round 2, that is, a round-2 write
carrying that state's true value. Qwen3-8B gets more games because its late holders read that title about half the
time (Gemma's 92%) and the effect forecast for it is smaller. Intervals resample seeds, jointly across the arms of a
model. Analysis: `scripts/analyze_plant.py`, written before the runs; on the base pairs (`gemma_n2`, `main_n2`) it gives
the arm-0 rates below.

## Forecast if the notes cause the posting

The observational rates on the same title, pooled over every swarm size of this study and the aaai branch (late holders
that read the favourite title in round 1, by the number of other copies' notes on it; n in brackets):

| Notes on the page | 0 | 1 | 2–3 | 4–7 | 8–15 | 16–31 | 32–63 | 64+ |
|---|---|---|---|---|---|---|---|---|
| Gemma, % | 29.7 (37) | 17.8 (90) | 23.6 (174) | 22.9 (389) | 26.7 (744) | 40.9 (1,554) | 68.0 (3,005) | 70.2 (2,784) |
| Qwen3-8B, % | 90.8 (65) | 68.4 (98) | 70.5 (237) | 77.4 (446) | 82.3 (824) | 82.7 (1,190) | 87.1 (1,520) | 81.8 (1,072) |

In a planted pair a late holder on that title sees k planted notes and at most one from the other copy, so:

| Arm | Gemma, % | Qwen3-8B, % |
|---|---|---|
| 0 | **16** (the base pairs: 15 of 92) | **80** (41 of 51) |
| 4 | **23** | **77** |
| 16 | **41** | **83** |
| 32 | **68** | **87** |

If the notes do not cause it (the observational curve reflects which copies end up on crowded pages, not what they
read there), every arm stays near arm 0.

**Registered:**
- (a) Gemma: posting in arm 32 exceeds arm 0 by at least 20 points, and the 95% interval of the difference lies
  above zero.
- (b) Gemma: arms 16 and 32 are each within 15 points of the observational rates for their page lengths
  (40.9% and 68.0%).
- (c) Qwen3-8B: posting in arm 32 is above arm 4.

Reported, not registered: arm "16, no number" against arm 16 (do the notes act through the answer they carry, or
through being there); the same rates over all late holders, whatever they read; posting in round 4; how often a late
holder answers question 1 with the planted value; the collusion rate (planted values never count as a use).
