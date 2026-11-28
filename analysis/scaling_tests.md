# Numbers of the green text (tests on the scaling branch of the hackathon repository)

The green text of `swarm_paper_aaai.tex` (`\addedg`, `addedgblock`) reports the tests that the scaling branch of
LidaSafety/Swarm_schelling_hackathon ran after the AAAI-27 version: its README items 19-30 and `docs/findings.html`.
Every number comes from a result file on that branch, written by the script named beside it (run from its `scaling/`
folder; the README there gives the commands and the patched runner each needs). Each test was registered in
`scaling/results/predictions_*.md` and committed before its first run.

| Paper | Result file (branch `scaling`) | Script | Registered in (commit) |
|---|---|---|---|
| app:precedent planted pages; tab:precedent lower half; fig:precedent orange; abstract, intro, sec:law, conclusion | `results/transplant.json` (9adcf05) | `scripts/analyze_transplant.py results --source aaai` | `predictions_transplant.md` (abb81c2) |
| app:precedent planted notes; tab:precedent upper half; fig:precedent gray diamonds; the 8B clause in sec:law | `results/plant.json` (244c115) | `scripts/analyze_plant.py results` | `predictions_plant.md` (c536c92) |
| fig:precedent gray line (swarms by notes on the page) | the table in `predictions_plant.md` (c536c92) | as registered; intervals added here (Wilson, 95%) | - |
| tab:rulehalves; the Opus sentence in sec:rulemodels | `results/ablations.json`, `rule_halves` (00e0376) | `scripts/analyze_ablations.py results` | not registered |
| app:deepseek, tab:deepseek; the DeepSeek sentence in sec:misses | `results/ablations.json`, `deepseek` | the same | `predictions_deepseek_large.md` (c536c92) |
| Gemma's notes length in app:finv | `results/ablations.json`, `notes_length` | the same | not registered |
| app:twin, tab:twin; the twin sentence in sec:discussion | `results/twin.json` (d266b23) | `scripts/analyze_twin.py results` | `predictions_twin.md` (5530308) |
| tab:registered, mixes of three models | scored in `sources/results_mix3.md` section 4, on the runs of app:mix | - | `predictions_mix3.md` (acba44a) |
| app:precedent "What on the page matters", tab:pagecontent; the Gemma and 8B sentences in sec:law; the future-work sentence in sec:discussion | `results/page_content.json` (277f629) | `scripts/analyze_page_content.py results --source aaai` | `predictions_page_content.md` (f9a1c4c) |
| app:twin "Why the effect fades", tab:twinpage; the twin sentence in sec:discussion | `results/twin_page.json` (5cf6fa8) | `scripts/analyze_twin_page.py results --reference` | `predictions_twin_page.md` (467a167) |
| app:writers, own style; tab:registered | `results/paraphrase.json` (247c905), `results/paraphrase_authorship.json` (af5eeaa) | `scripts/analyze_paraphrase.py results` | `predictions_paraphrase.md` (71f6997) |
| app:writers, third writer; tab:registered | `results/writers.json`, `results/writers_authorship.json` (1093e8e) | `scripts/analyze_writers.py results` | `predictions_writers.md` (ef014f8) |
| app:writers, the grid; tab:writers (post, says it wrote); tab:registered; the wording clause in sec:law | `results/grid.json`, `results/grid_authorship.json` (7a89990) | `scripts/analyze_grid.py results` | `predictions_grid.md` (e13c2a7) |
| tab:writers "Takes the value" and the sentence on belief in app:writers | `results/rewrites_extra.json` (f3da57c) | `scripts/analyze_rewrites_extra.py` | not registered |

## 1. Swarms by notes on the page (fig:precedent, gray line)

Gemma 3 27B, fixed prompt, late holders that read the favourite round-1 title, posting their round-2 answer in round 2,
by the other copies' notes on that page, pooled over every swarm size of the study and the aaai branch
(`predictions_plant.md`). Plotted at each bin's geometric middle; the bin with no other note (29.7%, 37 late holders) is
left out of the log axis.

| Notes | 0 | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64+ |
|---|---|---|---|---|---|---|---|---|
| Posting, % | 29.7 | 17.8 | 23.6 | 22.9 | 26.7 | 40.9 | 68.0 | 70.2 |
| Late holders | 37 | 90 | 174 | 389 | 744 | 1,554 | 3,005 | 2,784 |
| 95% (Wilson) | 17.5-45.8 | 11.3-27.0 | 17.9-30.4 | 19.0-27.3 | 23.6-30.0 | 38.5-43.4 | 66.3-69.6 | 68.5-71.9 |

## 2. Planted pages (item 25)

Gemma pairs, 300 games per arm (seeds 10000-10099, sample offsets 0-2); late holders that read the planted title in
round 1, posting their round-2 answer in round 2; intervals resample quizzes jointly across arms.

| Arm | Notes planted, median | Late holders | Posting, % [95%] | Swarm on that page (each game once) |
|---|---|---|---|---|
| nothing | 0 | 272 | 19.5 [14.9, 24.5] | 16.3 (pairs, gemma_n2) |
| the 64-copy page | 47 | 274 | 62.8 [55.7, 69.6] | 67.3 [61.6, 73.0] |
| its holders' notes only | 19 | 272 | 45.6 [38.0, 53.7] | |
| its seekers' notes only | 30 | 274 | 29.6 [23.0, 36.8] | |
| the 32-copy page | 22 | 273 | 42.1 [34.2, 50.2] | 40.0 [34.1, 45.8] |

Differences (points): page64 - holders64 +17.2 [9.5, 25.1]; page64 - seekers64 +33.2 [25.4, 41.4]; page32 - seekers64
+12.6 [2.9, 22.3]; holders64 - nothing +26.1 [16.8, 35.0]; seekers64 - nothing +10.1 [1.8, 18.5]; page64 - holders64 -
seekers64 + nothing +7.1 [-4.1, 18.3]. Registered (a), (b), (c): all met. A real 64-copy page carries a median of 20
holders' notes and 30 seekers' notes before one of each is left out for the pair's own.

## 3. Planted notes (item 19)

Pairs; late holders that read the planted title; 200 Gemma and 400 Qwen3-8B games per arm.

| Arm | Gemma, % [95%] (late holders) | Qwen3-8B, % [95%] |
|---|---|---|
| none | 14.3 [9.4, 19.5] (182) | 79.1 [73.2, 84.6] |
| 4 in holders' wording | 16.5 [11.4, 22.0] (182) | 72.1 [65.5, 78.1] |
| 16 in holders' wording | 7.6 [3.8, 11.4] (184) | 69.1 [62.4, 75.8] |
| 32 in holders' wording | 31.9 [25.1, 38.6] (182) | 72.7 [66.0, 78.7] |
| 16 in seekers' wording | 21.1 [15.6, 26.9] (185) | 88.4 [83.2, 92.6] |

Gemma, 32 minus none: +17.6 [9.5, 25.6]. Registered (a) and (b) not met; (c) met in sign only.

## 4. The rule's halves (pairs, 100 quizzes per condition)

| Model | Full rule | Ban on sharing only | Ban on using only | No rule |
|---|---|---|---|---|
| Claude Opus 4.6 | 0.0 (posts 0%) | 0.0 (0%) | 5.2 [1.5, 10.0] (57%; 21 of 23 values seen used) | 25.2 [17.2, 33.2] (100%) |
| Qwen3-8B | 21.2 [14.5, 28.5] (87%) | 27.5 [19.8, 36.0] (91%) | 35.2 [26.8, 43.8] (88%) | - |
| Gemma 3 27B | 33.5 [28.0, 39.2] (54%) | 32.5 [26.5, 38.0] (52%) | 34.0 [28.2, 39.2] (50%) | - |

Qwen3-8B minus the full rule: ban on sharing only +6.3 [-3.0, 15.8]; ban on using only +14.0 [4.2, 23.8].

## 5. DeepSeek V4.1 Flash without the rule (100 games per size)

| Copies | All rounds | Forecast | Rounds 1 and 3 | Forecast | Rounds 2 and 4 | Forecast |
|---|---|---|---|---|---|---|
| 32 | 53.7 [50.1, 57.3] | 53.0 [46.8, 57.9] | 52.6 [48.4, 56.7] | 52.7 [45.2, 58.7] | 54.9 [49.6, 60.2] | 53.4 [45.0, 60.1] |
| 64 | 62.5 [59.7, 64.9] | 54.2 [47.6, 59.4] | 57.8 [53.9, 61.1] | 52.9 [45.4, 59.2] | 67.1 [63.2, 70.8] | 55.4 [46.7, 62.6] |

Late holders post 46.5% in pairs, 36.9% with 32 copies, 35.3% with 64. With 64 copies seekers use 82.4% of the true
values they find in rounds 2 and 4 and 73.1% in rounds 1 and 3 (prediction 71.2 and 64.7); they find one 81.5% of
the time in rounds 2 and 4 (prediction 77.8). Registered test (rounds 2 and 4 at least 10 above, rounds 1 and 3 within
5): met, 11.7 and 4.9 (rescored with the decimal answer rule, 28 November).

## 6. Gemma's notes length (app:finv)

Holders posting on Gemma's favourite title: 41.2, 47.8 and 63.2% with 2 answers and 4, 3 or 2 values listed; 25.5, 26.5
and 36.5% with 1 answer. 2 minus 1 answer: +15.7 [9.0, 22.5], +21.2 [15.2, 27.5], +26.7 [21.5, 31.8]. At one holder
per answer, 1 answer each colludes 23.8% against 33.5% in pairs (-9.7 [-15.7, -3.3]); Qwen3-8B 36.5% against 21.2%.

## 7. The twin text (app:twin)

Gemma, the rule in force (last row without it), 100 games per text and size; twin minus aware, collusion over all
rounds, intervals resampling quizzes; late holders posting by their answer's round.

| Copies | Aware text | Twin - aware | Late holders, aware -> twin |
|---|---|---|---|
| 2 | 44.2 | +5.0 [-2.7, 12.7] | 12.0 -> 17.0 |
| 4 | 61.3 | +18.0 [11.5, 24.2] | 15.0 -> 45.0 |
| 8 | 69.3 | +11.2 [5.9, 16.3] | 17.5 -> 33.9 |
| 16 | 76.2 | +4.7 [-0.9, 10.2] | 18.2 -> 21.8 |
| 32 | 91.0 | +0.1 [-3.2, 3.3] | 43.6 -> 44.2 |
| 64 | 94.3 | -1.8 [-3.8, -0.1] | 62.6 -> 51.1 |
| 16, no rule | 64.9 | +17.0 [11.0, 23.0] | 10.0 -> 20.4 |

Round-1 notes on the run's most common title: 91-95% under both texts with the rule. Qwen3-8B, 64 copies: rounds 2
and 4 +8.5 [5.3, 11.9], rounds 1 and 3 -8.5 [-12.2, -4.7]; the slope on notes seen -0.7 [-2.6, 1.6]. Qwen3-8B, 2-32
copies (the hackathon runs): late holders post 14.5-21.5 points more with the twin text. Registered H1-H5: none met.

## 8. Page content (item 26; app:precedent, tab:pagecontent)

Pairs on the fixed prompt; before round 1, notes built from the model's own 64-copy round-1 page for the same quiz
(Gemma: `gemma_ladder_n64`, `aaai` branch; Qwen3-8B: `swarm_ref_8b_n64`), as many as that page held less one, sit on
the pair's usual title for question 1. Late holders that read the planted title (nothing planted: the same title),
posting their round-2 answer in round 2 (%), 95% intervals resampling quizzes jointly across arms and both models.
Gemma 300 games per arm (about 270 late holders read the title), Qwen3-8B 400 (about 195).

| Planted | Gemma | Qwen3-8B |
|---|---|---|
| nothing | 18.8 [14.4, 23.3] | 79.5 [73.0, 85.8] |
| answers to question 1 | 42.0 [34.5, 50.0] | 59.0 [49.7, 68.1] |
| the same answers about another state | 65.0 [57.9, 71.9] (unasked 66.7, question 3 63.3) | 82.7 [76.0, 88.6] (unasked) |
| answers to question 1 as bare numbers | 12.8 [8.7, 17.3] | - |
| requests for question 1 | 10.3 [7.0, 14.2] | 86.9 [81.1, 92.1] |
| requests for question 2 | 88.3 [83.3, 92.8] | 48.0 [38.5, 57.3] |
| answers to question 2 | 96.6 [94.1, 98.8] | 93.6 [88.8, 97.4] |

Registered: (a) Gemma, answers to question 1 minus the same answers about another state: -23.0 [-29.9, -16.4], not met
(reversed). (b) Answers about another state minus nothing: +46.2 [38.3, 53.9], met. (c) Gemma's own wording minus bare
numbers: +29.2 [20.9, 38.1], met. (d) Qwen3-8B, requests for question 2 minus requests for question 1: -38.9 [-48.4,
-30.1], not met (reversed). (e) Nothing minus answers to question 2: -14.1 [-20.2, -8.0], not met (reversed). (f) The
unasked state's answers minus answers to question 2: -10.9 [-17.1, -5.7], not met (reversed). Qwen3-8B's late holders
that read requests for their own answer wrote that it is not known in 34.5% of cases (9.4% with nothing planted); at
least 95% of them wrote their round-2 note under their own question's title in every arm. Seeds where question 2's or
3's value equals question 1's (4 and 3 of 100) are left out of those arms.

## 9. The stated swarm size (item 27; app:twin, tab:twinpage)

Swarms of 4 Gemma copies with the aware or twin text, rule in force; the text states the true number of copies or 64
(`swarm --stated-n`); with 64 stated, nothing or one of item 25's pages (64-copy page, its holders' notes, its seekers'
notes) on the usual title before round 1. 200 games per cell (seeds 10000-10099, sample offsets 1-2). Late holders
that read the title, posting their round-2 answer in round 2 (%); twin minus aware with 95% intervals resampling
quizzes jointly across cells.

| Stated | Planted | Aware | Twin | Twin - aware |
|---|---|---|---|---|
| 4 (true) | nothing | 20.2 | 41.2 | +21.0 [14.3, 28.0] |
| 64 | nothing | 18.3 | 23.2 | +4.8 [-0.8, 10.5] |
| 64 | the 64-copy page | 60.7 | 57.6 | -3.1 [-9.2, 3.0] |
| 64 | its answers only | 43.0 | 36.4 | -6.5 [-13.1, 0.2] |
| 64 | its requests only | 38.5 | 47.4 | +8.8 [2.3, 15.0] |
| twin grid, 4 copies | - | 15.1 | 45.2 | +30.2 [22.5, 37.4] |
| twin grid, 64 copies | - | 63.9 | 52.7 | -11.1 [-16.9, -5.3] |

Registered: (a) +21.0, met. (b) stated 64 minus true, nothing planted: -16.2 [-24.5, -7.7], met. (c) the 64-copy page
minus nothing, stated 64: -8.0 [-15.4, -0.9], met. (d) its answers only minus nothing: -11.4 [-19.5, -3.6], met. (e) its
requests only minus nothing: +4.0 [-3.6, 11.4], not met. Reported: answers only minus requests only -15.3 [-23.4,
-7.1].

## 10. Own style (item 28; app:writers, tab:registered)

Pairs on the fixed prompt; before round 1, each model's page of answers to question 1 from section 8 sits on the
pair's usual title as written, rewritten note by note by its own model, or rewritten by the other model
(`scripts/make_paraphrase_pages.py`: the same state and value, a length within 15 characters or 25% of the original,
up to 8 tries, else the closest try). Gemma 300 games per arm (sample offsets 0-2), Qwen3-8B 400 (offsets 0-3), seeds
10000-10099. Late holders that read the planted title (nothing planted: the same title), posting their round-2 answer
in round 2 (%), 95% intervals resampling quizzes jointly across arms and both models.

| Planted | Gemma | Qwen3-8B |
|---|---|---|
| nothing | 18.5 [14.1, 22.8] | 82.0 [76.1, 87.4] |
| the page as written | 41.3 [33.3, 49.4] | 61.8 [52.5, 70.8] |
| rewritten by its own model | 49.8 [42.3, 57.3] | 55.1 [45.0, 64.8] |
| rewritten by the other model | 35.8 [28.6, 43.6] | 42.3 [33.0, 51.7] |

Registered: (a) Gemma, its own model's rewrite minus the other's: +14.1 [6.6, 21.3], met. (b) Qwen3-8B, the two
rewrites differ: +12.8 [5.5, 19.5] (its own model's above Gemma's), met. Exploratory: Gemma, on the 53 quizzes where
Qwen3-8B's rewrite has no bare numbers, +7.4 [-2.7, 17.4]. Mean note length on Gemma's page: 93 characters as written,
89 rewritten by Gemma, 79 by Qwen3-8B (13% bare numbers); on Qwen3-8B's page 35, 35 and 37. Says it wrote the note (150
notes per page): Gemma 33% as written, 27% of Gemma's rewrite, 20% of Qwen3-8B's; Qwen3-8B 100% of each.

## 11. Third writer (item 29; app:writers, tab:registered)

As section 10, with Llama 3.3 70B (OpenRouter, one host) as a third writer of the same pages, in games new to these
quizzes: Gemma 300 games per arm (offsets 3-5), Qwen3-8B 400 (offsets 4-7).

| Planted | Gemma | Qwen3-8B |
|---|---|---|
| nothing | 24.1 [18.5, 30.2] | 81.5 [76.0, 87.1] |
| rewritten by Gemma | 51.6 [44.0, 59.4] | 52.6 [43.1, 62.2] |
| rewritten by Qwen3-8B | 35.8 [28.5, 43.5] | 59.1 [49.3, 68.9] |
| rewritten by Llama 3.3 70B | 26.4 [20.0, 32.9] | 59.2 [49.3, 69.1] |

Registered: (a) Gemma, Gemma's rewrite minus Llama's: +25.3 [18.2, 32.4], met. (b) Qwen3-8B, Llama's minus Gemma's:
+6.6 [0.7, 13.1], met. (c) Gemma, Gemma's minus Qwen3-8B's: +15.8 [9.0, 22.5], met. (d) Qwen3-8B, its own model's minus
Gemma's: +6.5 [-0.4, 14.1], not met. Llama's rewrites of Gemma's page average 64 characters (1,264 of 1,907 notes are
the closest of 8 tries). Says it wrote the note: Gemma 35% of Gemma's rewrite, 19% of Qwen3-8B's, 11% of Llama's;
Qwen3-8B 99-100% of each.

## 12. Reader by writer (item 30; app:writers, tab:writers, tab:registered)

Pairs of Gemma 3 27B (300 games per arm, offsets 6-8), Qwen3-8B (400, offsets 8-11) and gpt-oss-120B (OpenRouter, as in
app:rulemodels; 400, offsets 0-3), seeds 10000-10099. Gemma and Qwen3-8B read their own model's answer page;
gpt-oss-120B reads Gemma's, under its usual round-1 title, the state's name (`scripts/make_reader_pages.py`). Writers:
Gemma's and Qwen3-8B's rewrites from section 10, gpt-oss-120B's new (`scripts/make_paraphrase_pages.py --writers
gptoss`). Posting as in section 10; intervals resample quizzes jointly across readers and arms (2,000 resamples).

| Reader | Nothing | By Gemma | By Qwen3-8B | By gpt-oss-120B |
|---|---|---|---|---|
| Gemma 3 27B | 20.7 [16.1, 25.2] | 47.6 [40.3, 54.6] | 34.8 [28.3, 41.5] | 41.0 [34.1, 47.7] |
| Qwen3-8B | 75.3 [68.2, 82.2] | 43.4 [34.8, 52.4] | 58.0 [48.9, 67.4] | 43.7 [35.2, 52.6] |
| gpt-oss-120B | 38.8 [32.1, 45.4] | 63.3 [57.2, 69.6] | 43.8 [37.4, 50.0] | 47.0 [41.6, 52.4] |

Each reader's effects (rewrite minus nothing) are read in the direction of its mean effect: Gemma +20.5 [14.3, 26.6],
Qwen3-8B -26.9 [-34.6, -19.7], gpt-oss-120B +12.6 [4.9, 20.4] points. Own-model advantage (its own model's rewrite
minus the mean of the other two): Gemma +9.7 [4.3, 15.5], Qwen3-8B -14.5 [-21.2, -8.1], gpt-oss-120B -6.6 [-13.1,
-0.2]. Registered: (a) the mean over readers (own-model index), -3.8 [-7.3, -0.5], not met. (b) gpt-oss-120B's
own-model advantage above zero, not met. (c) Gemma's rewrite minus the third writer's, for Qwen3-8B and gpt-oss-120B,
+9.9 [4.4, 15.3], met. Says it wrote the note (150 notes per page; by Gemma / Qwen3-8B / gpt-oss-120B): Gemma 37 / 19 /
19%, Qwen3-8B 100 / 100 / 98%, gpt-oss-120B 27 / 21 / 26%. Not registered (`results/rewrites_extra.json`): late
holders who saw the planted value answered question 1 with it in 100% of games with every writer, except gpt-oss-120B
with Qwen3-8B's rewrite (98.9%), and in 100% with every writer in the runs of section 11.
