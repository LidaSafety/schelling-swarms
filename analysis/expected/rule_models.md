# The rule ablation across models at N = 16

Quiz accuracy: % correct on the capability quiz (504 MMLU-Pro questions, each model with its game settings; a reply with no answer within 2048 tokens counts as wrong). CR in % with 95% intervals from resampling games; the difference is no rule less rule. Writes, reads: share of write and read steps that edit or view a page. Empty: share of seekers' answers left empty. Rerun with `python rule_models.py`.

| Model | Run on | Quiz accuracy | Games per arm | CR, rule | CR, no rule | Difference | Writes, rule → no rule | Reads, rule → no rule | Empty answers, rule → no rule |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | vLLM, bf16 | 58.5 [54.2, 62.7] | 100, 100 | 38.2 [34.2, 42.3] | 39.4 [35.8, 43.0] | +1.1 [-4.4, +6.3] | 97 → 90 | 100 → 100 | 0 → 0 |
| Qwen3-8B | vLLM, bf16 | 62.1 [57.9, 66.5] | 100, 100 | 61.6 [57.8, 65.2] | 78.3 [75.7, 80.9] | +16.8 [+12.2, +21.3] | 79 → 87 | 98 → 99 | 0 → 0 |
| Qwen3-14B | vLLM, bf16 | 68.3 [64.1, 72.2] | 100, 100 | 22.0 [18.8, 25.0] | 20.0 [17.7, 22.3] | -2.0 [-5.8, +2.0] | 68 → 79 | 100 → 100 | 0 → 0 |
| Qwen3-32B | vLLM, bf16 | 70.8 [67.1, 74.8] | 100, 100 | 3.1 [1.8, 4.7] | 20.2 [17.1, 23.5] | +17.1 [+13.4, +20.7] | 9 → 40 | 99 → 98 | 0 → 1 |
| Gemma 3 27B | vLLM, bf16 | 68.8 [65.1, 72.8] | 100, 100 | 73.2 [69.2, 77.2] | 61.8 [57.7, 65.8] | -11.4 [-17.1, -5.7] | 83 → 77 | 97 → 95 | 0 → 0 |
| Qwen3-32B | OpenRouter, DeepInfra fp8 | 70.2 [66.1, 74.2] | 100, 100 | 9.1 [6.7, 11.8] | 24.0 [21.0, 26.9] | +14.9 [+11.0, +18.8] | 19 → 48 | 98 → 98 | 0 → 1 |
| Qwen3-235B-A22B-2507 | OpenRouter, DeepInfra fp8 | 76.2 [72.6, 79.8] | 100, 100 | 21.3 [18.6, 24.2] | 46.1 [43.7, 48.5] | +24.8 [+21.0, +28.3] | 59 → 58 | 79 → 82 | 28 → 14 |
| gpt-oss-120B (low reasoning) | OpenRouter, Crusoe bf16 | 74.8 [70.8, 78.6] | 100, 100 | 31.8 [29.0, 34.5] | 27.3 [25.1, 29.8] | -4.5 [-8.0, -0.8] | 61 → 81 | 74 → 69 | 4 → 3 |
| GLM-5.3 Flash (low reasoning) | OpenRouter, Parasail fp8 | 83.7 [80.4, 86.9] | 100, 100 | 29.6 [26.2, 32.8] | 36.4 [32.9, 39.8] | +6.8 [+2.0, +11.7] | 27 → 72 | 60 → 92 | 2 → 1 |
| DeepSeek V4.1 Flash | OpenRouter, DeepInfra fp8 | 84.9 [81.7, 88.1] | 100, 100 | 6.9 [4.8, 9.1] | 43.3 [39.6, 46.9] | +36.4 [+32.2, +40.7] | 4 → 46 | 37 → 88 | 2 → 0 |
| Qwen3.5-27B | OpenRouter, DeepInfra fp8 | 77.8 [73.8, 81.2] | 100, 100 | 2.3 [1.5, 3.3] | 24.3 [22.0, 26.6] | +22.0 [+19.6, +24.5] | 5 → 42 | 22 → 56 | 21 → 10 |
| Claude Haiku 4.5 | Anthropic API | 80.8 [77.2, 84.1] | 100, 100 | 4.5 [3.3, 5.8] | 20.5 [17.4, 23.6] | +16.0 [+12.8, +19.4] | 41 → 68 | 48 → 85 | 3 → 0 |
| Claude Haiku 5.5 | Anthropic API | 83.9 [80.6, 87.1] | 100, 100 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | +0.0 [+0.0, +0.0] | 0 → 0 | 0 → 1 | 96 → 97 |
| Claude Sonnet 5.5 | Anthropic API | 88.5 [85.9, 91.1] | 100, 100 | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.2] | +0.1 [+0.0, +0.2] | 0 → 6 | 0 → 7 | 0 → 0 |
| Claude Opus 4.6 | Anthropic API, 50 games per arm | 89.1 [86.3, 91.7] | 50, 50 | 0.0 [0.0, 0.0] | 55.2 [49.8, 60.2] | +55.2 [+49.8, +60.2] | 0 → 25 | 19 → 53 | 0 → 0 |
