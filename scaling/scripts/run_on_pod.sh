#!/usr/bin/env bash
# `python -m scb swarm ...` against the pods started by scripts/pod.py (one per model), with the settings of the
# original runs (configs/models.yaml), e.g. thinking off for Qwen3-8B: `pod.py endpoints` hands scb each model's
# address, key and request options (SB_ENDPOINTS). Examples (the scaling pilot; a mixed swarm, which needs both pods):
#   scripts/run_on_pod.sh swarm --models qwen3-8b --channels wiki --agents 2 4 8 16 --log none --awareness aware \
#       --rules forbidden --listing none --n 5 --tag pilot
#   scripts/run_on_pod.sh swarm --mix qwen3-8b:2,gemma-3-27b:2 --awareness unaware --rules forbidden --listing none
set -euo pipefail
cd "$(dirname "$0")/.."
endpoints="$(python3 scripts/pod.py endpoints)"
eval "$endpoints"
exec .venv/bin/python -m scb "$@"
