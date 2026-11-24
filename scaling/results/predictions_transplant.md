# Forecasts for transplanted pages (which part of a crowded page makes copies post)

Written before these runs start, and committed before the first run.

## Why

Up to its round-2 write, a late holder's only information about the other copies is the page it read in round 1: the
prompt is the same at every swarm size, and its own round-1 note is written before it reads. In Gemma 3 27B swarms on
the fixed prompt, late holders that read the main round-1 page post their round-2 answer in round 2 16.3% of the time
in pairs (`gemma_n2`, the usual title), 40.9% with 32 copies and 68.2% with 64 (`gemma_ladder_n32` and `_n64` on the
`aaai` branch).
So the rise comes from what that page shows, and the question is which part of it.

The planted notes (item 19, `results/predictions_plant.md`) put up to 32 notes in holders' wording on that page and
raised posting from 14.3% to 31.9%, a third of the rise. A real page differs. With 64 copies it carries a median of 20
holders' notes ("According to my notes, the share of employed workers in construction occupations in Iowa is 8.7%.")
and 30 seekers' notes, written by the late holders before they read ("Unknown. Need to find data for Iowa."; 56 of the
2,936 late holders' round-1 notes on that page carried an answer they held). In item 19, 16 notes in seekers' wording raised
posting more than 16 in holders' wording. These runs plant real round-1 pages, and each kind of note on its own, to
find out whether a late holder posts because of the answers it sees, the requests it sees, both together, or the
number of notes.

## Runs

Pairs of Gemma 3 27B on the fixed prompt (unaware, rule in force, no page list, `--log none`), 4 questions, 2 answers
per copy, fixed blocks, temperature 1, 250 tokens per call, vLLM 0.32.0, served from `unsloth/gemma-3-27b-it` as for
`gemma_n2` and `plant_gemma`. Before round 1, the notes of one page sit on the pair's usual round-1 title for question
1, "⟨State⟩ Construction Employment Share". Planted notes are shown like any note but never count as another copy's.

The pages come from the 100 games of `gemma_ladder_n64` and `gemma_ladder_n32` (`aaai` branch `ba7ee33`; seeds
10000–10099, the same quizzes as the pairs). A game's page is every note written in round 1 under the title most of its
copies wrote under (the usual title in 95 and 96 of the games), in the order the game showed them. Holders' notes are
those of copies holding question 1; seekers' notes those of the other copies. The last holder's note and the last
seeker's note are left out of each page, because the pair's own two copies, one of each, add theirs, so a late holder
that reads the planted title sees the swarm's count and mix of notes. Each page was checked against the page text a
late holder of that game was shown (`scripts/make_transplant_pages.py`, files in `results/transplant/`).

| Arm | Planted before round 1 | Notes per page, median (range) | Holders' | Seekers' |
|---|---|---|---|---|
| 0 | nothing | 0 | 0 | 0 |
| page64 | the 64-copy game's round-1 page | 47 (30–60) | 19 | 30 |
| holders64 | its holders' notes only | 19 (7–29) | 19 | 0 |
| seekers64 | its seekers' notes only | 30 (9–31) | 0 | 30 |
| page32 | the 32-copy game's round-1 page | 22 (14–30) | 9 | 15 |

Each arm plays seeds 10000–10099 three times (sample offsets 0, 1 and 2: the same quiz and page, the model resampled),
so every arm has 300 games on the same quizzes, and each page arm uses the page of the swarm game with the same seed:
1,500 games and 36,000 model calls. Runner: `patches/plant-pages-runner.patch` (adds `swarm --plant-file` and
`--sample-offset`); runs: `scripts/run_transplant_pairs.sh`.

Outcome, as in item 19: the late holders (copy A) whose round-1 read was the planted title (in arm 0, the same title):
the share that post their round-2 answer in round 2, that is, a round-2 write carrying that state's true value.
Intervals use 2000 resamples of the seeds, with all games of a seed, in every arm, moving together (they share the quiz
and the page). Analysis: `scripts/analyze_transplant.py`, written before the runs; on the source games it gives the
swarm rates below.

## Forecasts

The swarm rates on the same pages, each game counted once as each page is here (late holders that read the main
round-1 page; pooled over late holders in brackets), and the pairs on the usual title (`scripts/analyze_transplant.py
results --source aaai`, with the `aaai` folder as for the pages):

| Swarm | Late holders | Post their round-2 answer in round 2, % [95%] |
|---|---|---|
| 64 copies | 2,936 | **67.3** [61.6, 73.0] (68.2) |
| 32 copies | 1,476 | **40.0** [34.1, 45.8] (40.9) |
| Pairs (`gemma_n2`) | 92 | 16.3 [8.7, 23.9] |
| Pairs, item 19's arm without a plant (`plant_gemma`) | 182 | 14.3 [9.3, 19.2] |

If planting a page acts like reading it in a swarm, page64 and page32 land near 67.3% and 40.0%. For the parts, item
19 gives one guide: holders' notes alone gave 7.6% with 16 notes and 31.9% with 32, and 16 seekers' notes gave 21.1%.
If late holders post because a page shows copies both asking and answering, holders64 and seekers64 stay near those
rates and page32 beats seekers64, although it carries fewer notes. If the number of notes is what counts, seekers64,
with more notes than page32, posts at least as often as page32. If seekers' notes alone do it, seekers64 comes close
to page64.

**Registered:**
- (a) Planted pages act like the swarm's own: page64 is within 10 points of 67.3%, and page32 within 10 points of
  40.0%.
- (b) Neither kind of note alone gives what the whole page gives: page64 exceeds both holders64 and seekers64, each
  95% interval of the difference above zero.
- (c) Asking and answering together beat more notes of one kind: page32 exceeds seekers64, the 95% interval of the
  difference above zero.

Readings:
- **(a) not met:** a planted page does not act like reading it in a swarm, and (b) and (c) do not explain the
  swarm's rise.
- **(a), (b) and (c) met:** late holders post because a page shows copies both asking and answering; neither kind of
  note, nor the number of notes, explains it alone.
- **(a) and (b) met, (c) not:** each kind adds less than the whole page, and the number of notes also matters.
- **(a) met, (b) not, with seekers64 near page64:** the seekers' notes alone carry the rise.

Reported, not registered: holders64 and seekers64 against arm 0, and whether the two kinds add up (page64 − holders64
− seekers64 + arm 0); the same rates over all late holders, whatever they read; posting in round 4; within each page
arm, posting against the number of notes on the page; how often a late holder answers question 1 with a planted value
(planted values never count as collusion). Not covered: Qwen3-8B, whose late holders post 76–81% of their answers by
the answer's round at every size from 2 to 32 copies on this prompt (`results/twin.json`), and pages of other rounds.
