# Qwen3-8B seekers that opt out (sec:misses, app:lateholders, tab:optout)

Numbers for the 8B seeker opt-outs, 24 November 2026. Computed from the raw run files of the hackathon repository
(unaware prompt, rule in force, the same cells as fig:law) by three scripts in the working notes:
`notes/skip_audit.py` (what a skipped read is in the transcripts), `notes/seeker_skip.py` (prediction with the
opt-out term) and `notes/check_8b_terms.py` (independent re-check, oracle and holder-side variants).

## Definitions
- Opt out: a seeker has no read record in the round of the question it seeks. In the transcripts this is always an
  explicit `{"view": null}` (offered by the prompt, scb/deaddrop.py line 97); the harness answers "Nothing read."
  (scb/swarm.py lines 243-244). In pairs all 23 opted-out seekers also wrote `{"edit": null}`.
- Group A: copies that hold Q1 and Q3 and seek in rounds 2 and 4. Group B: hold Q2 and Q4, seek in rounds 1 and 3.
- "No other copy's note": the group-A copy's round-1 read showed no note by another copy (empty `authors`). Every
  group-A copy read a page in round 1.

## Round 2 opt-outs of group-A seekers (round 4 identical in pairs; group B never opts out)
| Copies | No other copy's note | Another copy's note | All |
|---|---|---|---|
| 2 | 23 of 72 | 0 of 28 | 23 of 100 |
| 4 | 22 of 89 | 7 of 111 | 29 of 200 |
| 8 | 29 of 113 | 20 of 287 | 49 of 400 |
| 16 | 23 of 120 | 30 of 680 | 53 of 800 |
| 32 | 22 of 112 | 72 of 1488 | 94 of 1600 |
| 64 | 12 of 71 | 73 of 3129 | 85 of 3200 |
| 128 | 1 of 13 | 39 of 3187 | 40 of 3200 |

Other models (pairs and 16 copies): no seeker opts out in Qwen3-4B, 14B, 32B or Gemma 3 27B.

## Predicted collusion rate, all rounds (%)
| Copies | Prediction from pairs | + opt-out share measured at N | + opt-out share predicted from pairs | Measured |
|---|---|---|---|---|
| 64 | 69.9 | 78.0 | 78.7 | 78.3 |
| 128 | 75.3 | 85.2 | 85.5 | 86.5 [83.3, 89.1] |

The pair prediction keeps the 23 pair opt-outs (coverage 0); the term removes them and applies 1 - P(opt out at N),
so nothing is counted twice. Predicted opt-out share from pairs vs measured: 1.0% vs 2.7% at 64, 0.4% vs 1.2% at 128.

## Late holders (holder side)
Post after a page with notes: 15 of 28 in pairs, 2504 of 3043 at 128. Letting each late holder's posting depend on
whether its round-1 page had another copy's note, at the rates measured at each N, moves the predicted collusion rate
by -0.6 to +0.2 points.
