# The title match of a swarm that mixes any number of models

One title distribution per model, p^m: the round-1 titles its copies write to, holders and seekers alike, on its pure swarms of N ≤ 8 (Qwen3-8B from the main runs, Gemma 3 27B from its own, the other Qwen3 sizes from the size comparison). State names are replaced by a placeholder, as in the paper. Rerun with `python mixtures.py`.

## 1. The title match M of the five models

| | 4B | 8B | 14B | 32B | Gemma |
|---|---|---|---|---|---|
| 4B | 0.335 | 0.027 | 0.158 | 0.120 | 0.047 |
| 8B | 0.027 | 0.303 | 0.047 | 0.039 | 0.054 |
| 14B | 0.158 | 0.047 | 0.212 | 0.211 | 0.192 |
| 32B | 0.120 | 0.039 | 0.211 | 0.361 | 0.450 |
| Gemma | 0.047 | 0.054 | 0.192 | 0.450 | 0.640 |
| Titles counted | 1,378 | 1,330 | 927 | 189 | 1,396 |

Qwen3-8B and Gemma 3 27B: 0.303, 0.054, 0.640; the paper's Table 2 has 0.30, 0.054 and 0.64.

Squared distance d² = M[a][a] + M[b][b] − 2 M[a][b] for each pair; an even mix of the two sags by d²/4:

| | 4B | 8B | 14B | 32B | Gemma |
|---|---|---|---|---|---|
| 4B | – | 0.584 | 0.231 | 0.456 | 0.881 |
| 8B | 0.584 | – | 0.421 | 0.587 | 0.835 |
| 14B | 0.231 | 0.421 | – | 0.152 | 0.468 |
| 32B | 0.456 | 0.587 | 0.152 | – | 0.101 |
| Gemma | 0.881 | 0.835 | 0.468 | 0.101 | – |

## 2. The four mixed swarms of the paper (Qwen3-8B with Gemma 3 27B)

*Pooled*: any two round-1 titles of the mix, without a title paired with itself. *Same game*: round-1 titles of two different copies of one game. f(x) predicts the first, f_N(x) = f(x) − gap/(N − 1) the second.

| Swarm | Gemma share x | f(x) | Measured, pooled | f_N(x) | Measured, same game | Weighted average |
|---|---|---|---|---|---|---|
| 2Q+2G | 0.5 | 0.262 | 0.257 | 0.193 | 0.195 | 0.471 |
| 6Q+2G | 0.25 | 0.230 | 0.229 | 0.208 | 0.219 | 0.387 |
| 4Q+4G | 0.5 | 0.262 | 0.264 | 0.233 | 0.239 | 0.471 |
| 2Q+6G | 0.75 | 0.399 | 0.400 | 0.377 | 0.392 | 0.555 |

Largest difference between prediction and measurement: 0.015.

## 3. Three models: Qwen3-4B, Qwen3-8B and Gemma 3 27B (the figure of Appendix F)

Corners (pure swarms): 0.335, 0.303, 0.640. Squared distances: 4B–8B 0.584, 4B–Gemma 0.881, 8B–Gemma 0.835.

Even three-way mix: f = 0.170 against a weighted average of 0.426; the gap is 0.256 = (sum of the three d²)/9 = 0.256.

Lowest title match over all mixes of the three: 0.152 at shares 4B 0.40, 8B 0.43, Gemma 0.17. The lowest over mixes of two of them is 0.172.

## 4. Check of the identities

On 2,000 random sets of 2 to 6 distributions over 2 to 11 titles, the largest violation of f = ||p̄||² and gap = variance = half the weighted sum of squared distances is 3.1e-16.
