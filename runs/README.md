# Runs that are not in `scaling/`

`scaling/results`, the three `scaling/aaai*` folders and `wiki_game/data/swarm4round` hold the runs from the hackathon
repository. The folders here
hold the rest of the runs the paper uses. They are kept apart because the scripts in `scaling/scripts` read every
`swarm*` folder they are given: putting the planted-note and N = 8 control folders in `scaling/results` would add
their cells to stage-1 cells that use the same seeds.

| Folder | What it is | Benchmark entry (`configs/experiments.yaml`) | Used for |
|---|---|---|---|
| `swarm_mix3/` | 22 swarms of 16 copies that mix Qwen3-4B, Qwen3-8B and Gemma 3 27B (pod J, 22 November 2026), one raw file per mix | 26 | fig:simplex, app:mix |
| `swarm_plant/` | Qwen3 swarms of 16 copies with a planted first note (true or wrong value), beside the same games without a plant (pod H, run 18) | 18 | The planted note in sec:misses |
| `swarm_scale_n8/`, `swarm_scale1_n8/` | Qwen3-4B, 8B, 14B and 32B at N = 8 with 2 answers per copy, and Qwen3-8B with 1 answer, rerun on pod H's GPU before its N = 64 and 128 runs, as a check against the earlier N = 8 runs | 17 | The N = 8 controls in `analysis/results_pod5.md` |
| `swarm_gemma_ladder_n8/` | Gemma 3 27B alone at N = 8, rerun on pod I's GPU as a check against the stage-1 run | 21 | The N = 8 control in `analysis/larger_n.py` |

Each folder has `summary.json` and `raw/<model>.jsonl.gz`, one record per game, as the benchmark writes them.

## `launch/`

The scripts that started the runs, and the package versions on each machine (`pip_freeze*.txt`, `gpu.txt`), without
the logs. `podA`: the 4-round reference runs (`wiki_game/data/swarm4round`). `podD`: the rule ablation of Qwen3-14B and 32B.
`podH`: N = 64 and 128, the planted note and the N = 8 controls (entries 17-19). `podI`: Gemma alone at N = 8-128, the
N = 16 mixes (`mix16.sh`, named in `analysis/predictions_mix16.md`) and the capability quiz. `podJ`: the three-model
mixes (`mix3.sh`, named in `analysis/predictions_mix3.md`). `claude/`, `openrouter/`, `capability_api.sh`: the runs
through the Anthropic API and OpenRouter. The run definitions themselves are in `configs/experiments.yaml`.
