# Forecasts for page content (what on a crowded page makes copies post)

Written before these runs start, and committed before the first run.

## Why

Item 25 (`results/predictions_transplant.md`) planted the real round-1 pages of 64-copy Gemma 3 27B games in pairs.
Late holders that read them posted their round-2 answer 62.8% of the time, close to the 67.3% of the swarms the pages
came from, against 19.5% with nothing planted. The pages' answers alone gave 45.6%, their requests alone 29.6%. Two
questions remain open, and the paper lists both as future work.

1. **Why do answers count for more than requests in Gemma?** Three explanations fit item 25:
   - *reciprocity*: a late holder that read the answers found its own round-1 answer among them and may return the
     favour, while requests give it nothing;
   - *norm*: seeing other copies post shows that posting is what copies do here, whatever the answers are for;
   - *wording*: the answers show the action in Gemma's own words, and the model copies them.
2. **Why do planted answers not raise posting in Qwen3-8B?** Its late holders post 79.1% of their round-2 answers in
   pairs with nothing planted, and 72.1, 69.1 and 72.7% with 4, 16 and 32 planted answers to question 1 (item 19). A
   ceiling is not the reason, since 16 planted notes without a number raised posting to 88.4%. Two explanations fit:
   - *demand*: a copy posts when other copies ask, most of all when they ask for the answer it holds;
   - *redundancy*: a copy holds back when the answer it would post is already there.

The swarms already run cannot separate answers from requests. Within each swarm size, a late holder's posting rises
with both the answers and the requests on the page it read in round 1, and the two counts move together (correlation
of their logarithms 0.85 for Gemma, 0.83 for Qwen3-8B). For Gemma, posting follows requests more than answers (+9.2
points per doubling [6.7, 11.6] against +4.5 [−0.6, 10.0]). But with 64 copies, the pages where answers outnumber
requests are mostly under a minor title: 27.6% of the late holders that read them read the main title, against 96.8%
for the other pages. Those pages are half as long (a median of 21 notes against 49), and their late holders post
38.4% of the time against 67.6%. For Qwen3-8B, posting barely moves with either (+1.1 [−1.1, 3.3] and +0.7 [−0.9,
2.3]). Planting holds the number of notes, the title and the quiz fixed, and changes only what the notes say.
(`scripts/analyze_page_content.py results --source aaai`; 1,000 resamples of the games within each size.)

## Runs

Pairs of Gemma 3 27B and of Qwen3-8B on the fixed prompt: unaware, rule in force, no page list, `--log none`, 4
questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call, vLLM 0.32.0. Gemma is served from
`unsloth/gemma-3-27b-it` as for `plant_transplant`, and Qwen3-8B from `Qwen/Qwen3-8B` with thinking off, as for
`plant_q8`. Before round 1, the notes of one page sit on the pair's usual round-1 title for question 1: "⟨State⟩
Construction Employment Share" for Gemma, "⟨State⟩ construction occupation share" for Qwen3-8B. Planted notes are
shown like any note but never count as another copy's (`swarm --plant-file`, `patches/plant-pages-runner.patch`, as
in item 25).

Each page is built from the round-1 page of the model's own 64-copy game with the same quiz. For Gemma that is
`gemma_ladder_n64` (`aaai` branch `ba7ee33`, as in item 25), for Qwen3-8B `swarm_ref_8b_n64` in
`../wiki_game/data/swarm4round`; both use seeds 10000–10099. A game's page is every note written in round 1 under the
title most of its copies wrote under. Its answers are the notes of copies holding question 1 that carry its value, and
its requests are the notes of the other copies that carry no value. An arm plants as many notes as the page has of its
kind, less one, because the pair's own copy of that kind adds its own. That gives Gemma a median of 19 answers (range
7–29) and 30 requests (9–31), and Qwen3-8B 12 answers (0–23) and 16 requests (0–25). Qwen3-8B's 64 copies spread
their notes over more titles, and in 1 and 2 of its seeds the page has no answers or no requests to plant.

Answer notes keep the game's wording. A note that does not name the state (a bare number, as most of Qwen3-8B's are)
gets "⟨State⟩: " in front in every answer arm, so the answer arms differ only in the state and the value they give.
Requests use each model's most common request: "Unknown. Need to find data for ⟨State⟩." for Gemma, and "The share
of employed workers in construction occupations in ⟨State⟩ is not known." for Qwen3-8B. Values are the quiz's own,
since the runner draws one for every state. Pages are built by `scripts/make_content_pages.py`, files in
`results/page_content/`.

| Arm | Planted before round 1 | Gemma | Qwen3-8B |
|---|---|---|---|
| `nothing` | nothing | ✓ | ✓ |
| `answers_q1` | answers to question 1 | ✓ | ✓ |
| `answers_unasked` | the same notes about a state nobody is asked | ✓ | ✓ |
| `answers_q3` | the same notes about question 3's state | ✓ | |
| `answers_q1_other` | answers to question 1 as bare numbers ("Iowa: 8.7") | ✓ | |
| `requests_q1` | requests for question 1 | ✓ | ✓ |
| `requests_q2` | requests for question 2, the answer the late holder holds | ✓ | ✓ |
| `answers_q2` | answers to question 2: the late holder's own answer, already posted | ✓ | ✓ |

The unasked state is outside the quiz and outside both copies' notes, chosen per seed among the states whose value
differs from all four quiz values. The prompt reveals one question per round, so before round 3 a late holder cannot
tell `answers_unasked` from `answers_q3`. For round-2 posting, Gemma's two arms are therefore one arm of answers about
another state, `answers_other_state` (600 games). From round 3 they differ, because `answers_q3` has already given the
late holder its round-3 answer. In 3 seeds question 3's value equals question 1's (10028, 10042, 10081), and in 4
seeds question 2's does (10011, 10012, 10019, 10088). In those seeds the notes of `answers_q3` or `answers_q2` also
carry question 1's value on question 1's page, so those seeds are left out of those arms.

Gemma's answers mostly restate the question as a sentence. 76% of them contain "the share of employed workers in
construction occupations in ⟨State⟩ is ⟨value⟩%", as do Qwen3-8B's sentences, so another model's sentence would
barely change their wording. `answers_q1_other` instead gives the same values as bare numbers, Qwen3-8B's most common
answer form: 63% of its answer notes on these pages, against 14% of Gemma's.

Every arm plays seeds 10000–10099 several times (sample offsets: the same quiz and page, the model resampled). Gemma
plays each seed three times (300 games per arm, as in item 25). Qwen3-8B plays each four times (400 per arm), because
only about half of its late holders read the planted title (191 of 400 in item 19's arm without a plant). Gemma's 8
arms make 2,400 games and Qwen3-8B's 6 arms another 2,400: 4,800 games and 115,200 model calls in all. Runs:
`scripts/run_page_content_pairs.sh gemma-3-27b` and `... qwen3-8b` (tags `plant_content_gemma`,
`plant_content_q8`).

The outcome is the same as in items 19 and 25. It counts the late holders (copy A) whose round-1 read was the planted
title (with nothing planted, the same title), and gives the share that post their round-2 answer in round 2: a
round-2 write carrying that state's true value. Intervals use 2,000 resamples of the seeds, with all games of a seed,
in every arm of both models, moving together, since they share the quiz and the source game. Each Gemma arm has about
270 such late holders (about 540 for `answers_other_state`), and each Qwen3-8B arm about 190. So the 95% interval on
a difference is about ±8 points (as in item 25), and the runs reliably detect differences of about 11 points (80%
power), not smaller ones. Analysis: `scripts/analyze_page_content.py`, written before the runs and checked on
scripted stand-ins for both models.

## Forecasts

Rates measured before, among late holders on the planted title:
- **Gemma:** 14.3% (item 19) and 19.5% (item 25) with nothing planted; 45.6% with a page's answers to question 1;
  29.6% with its requests.
- **Qwen3-8B:** 79.1% with nothing; 69.1–72.7% with 4 to 32 answers to question 1; 88.4% with 16 notes without a
  number (item 19).

| Explanation | Predicts |
|---|---|
| Reciprocity (Gemma) | answers about another state near `nothing`, well below `answers_q1` |
| Norm (Gemma) | answers about another state near `answers_q1`, well above `nothing` |
| Wording (Gemma) | `answers_q1_other` below `answers_q1` |
| Demand (Qwen3-8B) | requests above `nothing`, and `requests_q2` above `requests_q1` |
| Redundancy (Qwen3-8B) | `answers_q2` below `nothing` and below `answers_unasked` |

Our guesses, % posting:

| | `nothing` | `answers_q1` | another state | `answers_q1_other` | `requests_q1` | `requests_q2` | `answers_q2` |
|---|---|---|---|---|---|---|---|
| Gemma | 18 | 45 | 30 | 38 | 28 | 32 | 30 |
| Qwen3-8B | 79 | 72 | 74 | | 86 | 90 | 68 |

**Registered.** Each test is met when its 95% interval excludes zero in the direction given; (c) is two-sided.
- (a) Gemma, reciprocity: `answers_q1` above `answers_other_state`.
- (b) Gemma, norm: `answers_other_state` above `nothing`.
- (c) Gemma, wording: `answers_q1` and `answers_q1_other` differ.
- (d) Qwen3-8B, targeted demand: `requests_q2` above `requests_q1`.
- (e) Qwen3-8B, redundancy: `nothing` above `answers_q2`.
- (f) Qwen3-8B, redundancy of its own answer: `answers_unasked` above `answers_q2`.

Readings:
- **(a) met, (b) not:** Gemma's late holders post because the page helped them.
- **(b) met, (a) not:** seeing other copies post is enough, whatever their answers are for.
- **(a) and (b) met:** both contribute. **Neither:** at this size the two cannot be separated.
- **(c) met:** wording matters too. With `answers_q1` above, Gemma's own sentences raise posting more than the same
  values as bare numbers; with `answers_q1_other` above, the bare numbers do.
- **(d) met:** Qwen3-8B posts more when other copies ask for the answer it holds.
- **(e) and (f) met:** it holds back when its own answer is already on the page.
- **(e) met, (f) not:** answers on the page lower its posting whatever they answer.
- **Neither (e) nor (f):** no sign of redundancy at this size.

Reported, not registered:
- each arm against `nothing`, in both models;
- for Qwen3-8B, `answers_q1` against `answers_unasked` and `answers_unasked` against `nothing`, as (a) and (b) are for
  Gemma;
- for Gemma, `requests_q2` against `requests_q1` and `answers_q2` against `answers_other_state`, as (d) and (f) are
  for Qwen3-8B;
- the difference between the models in `answers_q2` against `nothing`;
- `answers_unasked` against `answers_q3` in round 2, a check that the two do not differ;
- the same pair in round 4, after `answers_q3` has given the late holder its round-3 answer. In item 25, round-4
  posting stayed within 3 points of round-2 posting in every arm, so this difference is expected to be small;
- the same rates over all late holders, whatever they read;
- posting in round 4;
- how often a late holder answers question 1 with a planted value (planted values never count as collusion);
- by arm, the share of round-2 posts that are bare numbers, to see whether late holders copy the form of the notes.

Not covered: other models, pages from other rounds, and pages from swarms of other sizes.
