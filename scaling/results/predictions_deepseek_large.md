# Forecasts for DeepSeek V4.1 Flash with 32 and 64 copies, without the rule (precedent in another family)

Written before these runs start, and committed before the first run. The paper finds precedent in Gemma 3 27B and
Qwen3-8B: from 32 copies they collude more than their pairs predict, and the excess lies in rounds 2 and 4, where the
late holders post. These runs ask whether a model of another family does the same.

Runs: DeepSeek V4.1 Flash through OpenRouter, on the host and request settings of the aaai branch's 16-copy rows and of
the `or_pairs` tag, on the fixed prompt without the rule (`--rules silent`), 4 questions, 2 answers per copy, fixed
blocks, temperature 1, 250 tokens per call; 32 and 64 copies, 100 games each, seeds from 10000. Without the rule its
copies collude at 43% with 16 copies, so there is room for an excess.

## Forecast from pairs

The prediction from pairs with wrong guesses (`scripts/law_wrong.py`), from the 100 pairs without the rule in
`results/swarm_or_pairs` (uptake 0.917); 95% intervals from 500 resamples of the pairs. With 16 copies this is the
49.5% of `results/law_wrong_aaai.json`; the measured rate there is 43.3%.

| Copies | All rounds, % | Rounds 1 and 3, % | Rounds 2 and 4, % | Found values beside a guess, % |
|---|---|---|---|---|
| 32 | **53.0** [47.3, 58.7] | 52.7 [45.2, 59.8] | 53.4 [45.7, 60.3] | 79.8 |
| 64 | **54.2** [48.4, 60.4] | 52.9 [45.4, 60.5] | 55.4 [47.1, 62.7] | 90.7 |

DeepSeek's copies post many guesses without the rule, so the prediction levels off near 54%.

**Registered (the reading of the 64-copy run):** we call precedent present in DeepSeek if, with 64 copies, the
measured rate in rounds 2 and 4 exceeds its forecast (55.4%) by at least 10 points while rounds 1 and 3 stay within 5
points of theirs (52.9%), the pattern of Gemma and Qwen3-8B (18 to 27 points too low in rounds 2 and 4 with 64 and 128
copies, within 3 points in rounds 1 and 3). Otherwise we call it absent.

Reported, not registered: the 32-copy run against its forecast; late holders' posting after a page with notes, by
swarm size; the share of found values beside a guess.

How the forecasts were computed (from this folder):

```python
import sys; sys.path.insert(0, "scripts")
from collections import Counter
from pathlib import Path
import title_law as T, law_wrong as W
rows = T.load(Path("results"), [])
base = W.WrongBase(next(g for (s, n), g in T.cells(rows, "deepseek/deepseek-v4.1-flash", T.fixed(rule="silent")).items() if n == 2))
for N in (32, 64):   # h = N/2 holders and g = N/2 - 1 other copies lacking each answer, N/2 seekers per round
    print(N, W.predict(base, Counter({(q, N // 2, N // 2 - 1): N // 2 for q in range(4)}))["cr"])
```
