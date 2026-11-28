# Seekers' use of a found true value, by agreement on the page

Written by `agreement.py` (23 November) from the 64-copy games of the scaling branch of the hackathon repository
(commit a3709a3): `results/swarm_deepseek_large` (DeepSeek V4.1 Flash without the rule, the runs of app:deepseek)
and `results/swarm_scale_n64` (Qwen3-4B with the rule, the runs of fig:law). It reuses `notes_of`, `judge` and
`DEC` of `scripts/analyze_posting.py` unchanged. For each seeker chance: R = other copies whose notes on the pages
read by the question's round carry the true value; W = other copies whose notes carry a wrong value for it;
Wmax = the largest group of those copies that wrote the same wrong number. Used = the answer is a value found on
the wiki. Proportional = mean R/(R+W), what the prediction from pairs assumes beside wrong values (sec:law).
Rates pool chances; 95% intervals resample games (200 draws).

## DeepSeek V4.1 Flash, no rule, 64 copies (100 games)

Chances: 12,800; with a true value read: 10,271. R >= 1 agrees with the logged `seen` flag in 100.0% of chances. Beside at least one wrong value: 9,171 chances, with on average 6.5 true notes, 4.2 wrong notes, 3.5 distinct wrong values and a largest wrong group of 1.6; the true value is the most common value in 93.6% of them.

**Is the true value the most common value on the page?** (chances with R >= 1 and W >= 1)

| Page | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| True value most common (R > Wmax) | 8,582 | 79.9 [76.6, 82.5] | 62.2 |
| ... although wrong notes outnumber true ones (W > R) | 1,213 | 53.1 [41.5, 64.2] | 40.3 |
| Tied with a wrong value (R = Wmax) | 432 | 17.1 [10.0, 26.2] | 37.6 |
| A wrong value more common (R < Wmax) | 157 | 8.3 [0.6, 20.3] | 25.1 |

**By the true value's lead over the largest wrong group, R - Wmax** (chances with R >= 1 and W >= 1)

| R - Wmax | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| -2 or less | 26 | 0.0 [0.0, 0.0] | 21.9 |
| -1 | 131 | 9.9 [0.7, 22.2] | 25.7 |
| 0 | 432 | 17.1 [10.0, 26.2] | 37.6 |
| +1 | 764 | 37.3 [28.5, 46.4] | 48.7 |
| +2 to +3 | 2,051 | 74.1 [67.9, 79.9] | 55.5 |
| +4 or more | 5,767 | 87.6 [84.3, 89.7] | 66.3 |

**With no wrong value on the page, by true notes R**

| R | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| 1 | 201 | 84.6 [79.1, 90.9] | 100.0 |
| 2 | 189 | 93.7 [89.0, 96.9] | 100.0 |
| 3 or more | 710 | 99.3 [98.5, 100.0] | 100.0 |

**By round half** (all chances with R >= 1)

| Rounds | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| 1 and 3 | 5,056 | 73.1 [68.4, 77.0] | 60.1 |
| 2 and 4 | 5,215 | 82.4 [77.8, 87.1] | 69.0 |

## Qwen3-4B, rule in force, 64 copies (100 games)

Chances: 12,800; with a true value read: 10,437. R >= 1 agrees with the logged `seen` flag in 100.0% of chances. Beside at least one wrong value: 10,267 chances, with on average 12.9 true notes, 16.0 wrong notes, 7.4 distinct wrong values and a largest wrong group of 5.6; the true value is the most common value in 73.4% of them.

**Is the true value the most common value on the page?** (chances with R >= 1 and W >= 1)

| Page | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| True value most common (R > Wmax) | 7,539 | 59.5 [55.3, 63.3] | 51.8 |
| ... although wrong notes outnumber true ones (W > R) | 3,294 | 53.0 [45.5, 59.6] | 37.4 |
| Tied with a wrong value (R = Wmax) | 647 | 37.7 [26.5, 50.7] | 27.7 |
| A wrong value more common (R < Wmax) | 2,081 | 38.5 [28.8, 47.7] | 17.8 |

**By the true value's lead over the largest wrong group, R - Wmax** (chances with R >= 1 and W >= 1)

| R - Wmax | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| -2 or less | 1,540 | 39.5 [28.7, 50.0] | 16.4 |
| -1 | 541 | 35.7 [21.5, 47.8] | 21.6 |
| 0 | 647 | 37.7 [26.5, 50.7] | 27.7 |
| +1 | 479 | 40.9 [29.2, 53.2] | 34.2 |
| +2 to +3 | 855 | 48.7 [37.5, 57.8] | 37.0 |
| +4 or more | 6,205 | 62.5 [58.0, 66.8] | 55.1 |

**With no wrong value on the page, by true notes R**

| R | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| 1 | 54 | 77.8 [63.0, 90.7] | 100.0 |
| 2 | 22 | 63.6 [35.0, 90.0] | 100.0 |
| 3 or more | 94 | 87.2 [77.3, 95.7] | 100.0 |

**By round half** (all chances with R >= 1)

| Rounds | Chances | Used, % [95%] | Proportional, % |
|---|---|---|---|
| 1 and 3 | 5,538 | 67.9 [62.4, 72.9] | 41.5 |
| 2 and 4 | 4,899 | 39.0 [32.1, 44.5] | 47.4 |

## Reading (written by hand, not by the script)

- **DeepSeek follows the most common value.** Beside wrong values, its seekers use the true value far more often than
  the proportional rule when it is the most common value on the page (79.9% against 62.2%), and far less when a wrong
  value ties or leads (17.1% and 8.3%, against 37.6% and 25.1%). Use rises steeply with the true value's lead.
  Its guesses rarely agree with one another (4.2 wrong notes carry 3.5 distinct values), while holders all post the
  same true value, so the true value is the most common value in 93.6% of these chances. This is the mechanism that
  sec:misses gives for Qwen3-4B, and it accounts for DeepSeek's excess over the prediction coming from use, not from
  posting (app:deepseek). It acts in both round halves.
- **Qwen3-4B at 64 copies shows it only weakly.** Its use rises with the true value's lead (39% to 62%), but stays
  near 36-39% even when a wrong value is more common, well above the proportional rule (16-22%). Its evidence for
  favouring agreed values rests on the planted note at 16 copies (app:planted), not on this pattern.
- Caveats: rates pool chances (intervals resample games); wrong values are grouped by the exact decimals in the note,
  so the same wrong number written differently counts as two groups and Wmax is slightly understated.
