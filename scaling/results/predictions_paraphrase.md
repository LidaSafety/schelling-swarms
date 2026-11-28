# Forecasts for own-style notes (do copies respond more to answers written in their own model's style?)

Written before these runs start, and committed before the first run.

## Why

In item 26 (`results/predictions_page_content.md`), the wording of other copies' answers mattered for Gemma 3 27B.
Its late holders posted their round-2 answer 42.0% of the time after reading answers to question 1 in Gemma's own
sentences ("According to my notes, the share of employed workers in construction occupations in Iowa is 8.7%."),
and 12.8% after the same answers as bare numbers ("Iowa: 8.7"), Qwen3-8B's most common form. That comparison
changes two things at once: the form of the notes (a sentence or a number) and whose style they are in.

Language models can often recognize text that they or their own model wrote, and treat it differently. If copies
respond to other copies' answers partly because the answers read like their own, then answers in the copy's own
model's style should move it more than the same answers, of the same length, in another model's style. These runs
separate style from content and length by having each model rewrite the same answer page note by note.

## Runs

Pairs of Gemma 3 27B and of Qwen3-8B (thinking off) on the fixed prompt, as in item 26: unaware, rule in force, no
page list, `--log none`, 4 questions, 2 answers per copy, fixed blocks, temperature 1, 250 tokens per call, vLLM
0.32.0. Each model is served as for item 26. Before round 1, one page sits on the pair's usual round-1 title for
question 1. All pages hold the answers to question 1 from the model's own 64-copy game with the same quiz: item 26's
`answers_q1` page (`results/page_content/<model>_answers_q1.json`; Gemma a median of 19 notes, Qwen3-8B 12).

| Arm | Planted before round 1 |
|---|---|
| `nothing` | nothing |
| `own` | the answer page as written (item 26's arm, run again) |
| `self` | the same page, each note rewritten by the same model |
| `other` | the same page, each note rewritten by the other model |

The rewrites (`scripts/make_paraphrase_pages.py`) send each note alone to the writer: "Rewrite this note in your own
words, the way you would write it yourself. Keep its meaning, the state's name and the number exactly, and keep it
about the same length (about N characters)." The settings are temperature 0.7, up to 150 tokens, and each model's
serving settings. A rewrite must name the note's state and no other state, and carry the note's value and no other
number. It should be within max(15, 25%) characters of the original's length, and a rewrite that misses either is
asked for again, up to 8 times. After 8 tries the closest-length rewrite with the right content is kept; a note with
no such rewrite is left off its page. The rewritten pages are made after this registration is committed, with both
models served as for the runs, and committed before the first pair run, with their counts of kept, approximate and
dropped notes and their mean note lengths.

Manipulation check, reported, not registered (`scripts/check_authorship.py`): each reader is shown 150 notes drawn at
random from each of its three pages, one at a time, and asked whether it or a different AI model wrote the note.

Games: Gemma, 300 per arm (seeds 10000–10099, sample offsets 0–2); Qwen3-8B, 400 per arm (offsets 0–3), as in item
26. That is 2,800 games and 67,200 model calls, plus at most 8 rewrite calls per note. Runs:
`scripts/run_paraphrase_pairs.sh gemma-3-27b` and `... qwen3-8b` (tags `plant_style_gemma`, `plant_style_q8`).

The outcome is item 26's: among the late holders whose round-1 read was the planted title (with nothing planted, the
same title), the share that post their round-2 answer in round 2. Intervals use 2,000 resamples of the seeds, with
all games of a seed, in every arm of both readers, moving together. As in item 26, the 95% interval on a difference is
about ±8 points, so the runs reliably detect differences of about 11 points. Analysis:
`scripts/analyze_paraphrase.py`, written before the runs and checked end to end on scripted stand-ins.

## Forecasts

From item 26, late holders on the planted title: Gemma 18.8% with nothing planted, 42.0% with its own answer page,
12.8% with the same answers as bare numbers; Qwen3-8B 79.5% with nothing planted and 59.0% with its own answer page.

| Explanation | Predicts |
|---|---|
| Own style: copies respond more to their own model's style | Gemma: `self` above `other`. Qwen3-8B, whose own answers lowered its posting: `self` below `other` |
| Content only: the style of the notes does not matter | `self` and `other` close to each other and to `own`, in both models |
| Rewording: any rewrite weakens the page | `self` and `other` both between `own` and `nothing` |

Our guesses, % posting: Gemma 19 (`nothing`), 42 (`own`), 38 (`self`), 28 (`other`); Qwen3-8B 80, 59, 62, 70.

**Registered.** Each test is met when its 95% interval lies wholly on the stated side of zero.
- (a) Gemma: `self` above `other`.
- (b) Qwen3-8B: `self` and `other` differ.

Readings:
- **(a) met:** Gemma's late holders respond more to answers in Gemma's style than to the same answers, of matched
  content and length, in Qwen3-8B's style.
- **(b) met, with `self` below `other`:** Qwen3-8B's own-style answers lower its posting more than another model's do;
  **with `self` above `other`:** the reverse.
- **Neither:** no sign at this size that copies respond to their own model's style. Item 26's wording effect then comes
  from the form of the notes, not from whose style they are in.

Reported, not registered:
- `self` and `other` against `nothing`, and `own` against `self` (whether rewording itself matters);
- per page, the mean note length and the share of notes that are bare numbers;
- the manipulation check: per reader and page, the share of notes it says it wrote;
- the share of round-2 posts that are bare numbers, per arm.

Not covered: writers from other model families, requests, and rewriting every note inside a running swarm.
