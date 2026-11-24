# Forecasts for the twin text against a crowded page (Gemma, 4 copies told 64)

Written before these runs start, and committed before the first run.

## Why

In item 24 (`results/predictions_twin.md`), the twin text ("3 copies of you are taking this exact quiz ... the same
model, with the same weights, receiving these same instructions") raised how often Gemma 3 27B's late holders post
their round-2 answer in small swarms and lowered it in large ones. Among late holders that read the usual round-1
title (`scripts/analyze_twin_page.py results --reference`), it gave 45.2% against 15.1% for the aware text with 4
copies (+30.2 points [22.5, 37.4]). With 64 copies it gave 52.7% against 63.9% (−11.1 [−16.9, −5.3]).

Three things grow together with the swarm in those runs: the number of copies, the number the sentence states ("3
copies of you" or "63 copies of you"), and how crowded the page a late holder reads is. Gemma's copies meet on one
title so reliably that a late holder's page always holds about as many notes as the swarm has copies: 99.3–99.5% of
late holders read the usual title, with 4 copies and with 64. So the runs cannot separate the three. In Qwen3-8B,
whose copies spread over many titles, the twin text's effect was about the same on sparse and crowded pages (+16 to
+18 points with 32–64 copies; exploratory), but Qwen3-8B's posting barely follows the page.

Two explanations fit Gemma's pattern, and the second has two versions:
- *stated size*: being told that 63 copies take the quiz removes the boost, whatever the page shows;
- *crowded page*: a page full of other copies' notes overrides the twin text, through
  - its *answers*: other copies, under the twin text copies of itself, have already posted, so its own post is not
    needed;
  - or its *requests*.

These runs separate them in 4-copy swarms. The sentence states the true size or 64, and, told 64, a real 64-copy page,
its answers alone, its requests alone, or nothing is planted.

## Runs

Swarms of 4 Gemma 3 27B copies on the fixed prompt, but with the aware sentence or the twin sentence: rule in force,
no page list, `--log none`, 4 questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call, vLLM
0.32.0, served from `unsloth/gemma-3-27b-it` as for the twin grid. The sentence states how many other copies take the
quiz. With `stated_n` 64 it says "63 other instances of the same agent ..." or "63 copies of you ..." in place of 3
(`swarm --stated-n`, `patches/stated-size-runner.patch`, new). Nothing else in the prompt changes, and nothing else
gives the swarm's size. A late holder told 64 then reads, on the usual title, either the other 3 copies' notes, as on
a sparse page of a 64-copy swarm, or those notes after a planted page.

The planted pages are item 25's (`results/transplant/`, from `gemma_ladder_n64`, `aaai` branch `ba7ee33`):
- `page64`: the round-1 page of the 64-copy game with the same seed, a median of 47 notes;
- `holders64`: its holders' notes only, a median of 19;
- `seekers64`: its seekers' notes only, a median of 30.

The pages were built for pairs, so each has one note of each kind fewer than its source page. A 4-copy swarm adds up
to two of each, so a late holder sees about one note more than in the source game. The pages sit on the usual title
for question 1, "⟨State⟩ Construction Employment Share", which the late holders of the twin grid read in round 1.
Planted notes are shown like any note but never count as another copy's (`patches/plant-pages-runner.patch`).

| Told | Planted before round 1 | Aware | Twin |
|---|---|---|---|
| 4 (true) | nothing | ✓ | ✓ |
| 64 | nothing | ✓ | ✓ |
| 64 | `page64` | ✓ | ✓ |
| 64 | `holders64` | ✓ | ✓ |
| 64 | `seekers64` | ✓ | ✓ |

Every cell plays seeds 10000–10099 twice, with sample offsets 1 and 2 (offset 0 is the twin grid's own 4-copy runs).
That is 200 games per cell, 2,000 games and 96,000 model calls. Runs: `scripts/run_twin_page_grid.sh` (tags
`plant_twin_n4` and `plant_twin_told64`), on the runner at `76be3c7` with the seven earlier patches and
`stated-size-runner.patch`.

The outcome is the same as in items 24 and 25. It counts the late holders (with 4 copies, the two copies holding
questions 2 and 4) whose round-1 read was the planted title (with nothing planted, the same title), and gives the
share that post their round-2 answer in round 2. The twin effect in a condition is the twin cell's rate minus the
aware cell's. Intervals use 2,000 resamples of the seeds, with all games of a seed, in every cell, moving together.

Each cell has about 400 late holders. The 95% interval on a twin effect is about ±7 points, and on a difference
between two twin effects about ±10, so the runs reliably detect differences of about 14 points. Analysis:
`scripts/analyze_twin_page.py`, written before the runs and checked on a scripted stand-in.

## Forecasts

The rates these runs compare with, from the twin grid (sample offset 0, the same definitions):

| Swarm | Aware | Twin | Twin effect, points [95%] |
|---|---|---|---|
| 4 copies | 15.1% | 45.2% | +30.2 [22.5, 37.4] |
| 64 copies | 63.9% | 52.7% | −11.1 [−16.9, −5.3] |

| Explanation | Predicts |
|---|---|
| Stated size | told 64 with nothing planted, the twin effect near or below zero |
| Crowded page | told 64 with nothing planted, the twin effect stays near +30; with `page64` it falls toward −11 |
| Answers | the twin effect falls with `holders64` as it does with `page64` |
| Requests | the twin effect falls with `seekers64` as it does with `page64` |

Our guesses, % posting:

| Told | Planted | Aware | Twin |
|---|---|---|---|
| 4 | nothing | 15 | 45 |
| 64 | nothing | 15 | 35 |
| 64 | `page64` | 60 | 50 |
| 64 | `holders64` | 45 | 35 |
| 64 | `seekers64` | 30 | 40 |

**Registered.** Each test is met when its 95% interval lies wholly on the stated side of zero.
- (a) Replication: with the true size and nothing planted, the twin effect is above zero.
- (b) Stated size: told 64 with nothing planted, the twin effect is below the twin effect with the true size.
- (c) Crowded page: told 64, the twin effect with `page64` is below the twin effect with nothing planted.
- (d) Answers: told 64, the twin effect with `holders64` is below the twin effect with nothing planted.
- (e) Requests: told 64, the twin effect with `seekers64` is below the twin effect with nothing planted.

Readings:
- **(a) not met:** the twin text's boost does not replicate, and the other tests cannot be read.
- **(b) met, (c) not:** the number of copies the sentence states removes the boost; the page does not.
- **(c) met, (b) not:** the crowded page removes it; being told of many copies does not.
- **(b) and (c) met:** both contribute.
- **(d) met, (e) not:** the page's answers do it: copies that see other copies' answers hold back.
- **(e) met, (d) not:** the page's requests do it.
- **(d) and (e) met:** either kind of note does it.

Reported, not registered:
- the twin effect with `holders64` against `seekers64`;
- told 64 with `page64` against the 64-copy swarm: whether a 4-copy swarm told 64 and shown a 64-copy page posts as
  a 64-copy swarm does;
- the aware text's own response to the page (told 64, `page64` against nothing), beside item 25's pairs on the
  unaware text;
- the same rates over all late holders, whatever they read;
- posting in round 4;
- how often a late holder answers question 1 with a planted value (planted values never count as collusion).

Not covered: Qwen3-8B, whose twin effect did not shrink on crowded pages; other swarm sizes; the unaware text.
