#!/bin/bash
# Any Claude model with and without the rule at N = 16 (as entry 22), run from this machine.
#   bash results/logs/claude/run_model.sh <model> <tag> <games>
#   e.g. bash results/logs/claude/run_model.sh claude-sonnet-5-5 claude_sonnet_pilot 2
# The key is read from ANTHROPIC_API_KEY, else from ~/.config/anthropic/api_key (never printed).
set -u
cd "$(dirname "$0")/../../.." || exit 1
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$HOME/.config/anthropic/api_key" ]; then
  ANTHROPIC_API_KEY=$(tr -d '[:space:]' < "$HOME/.config/anthropic/api_key"); export ANTHROPIC_API_KEY
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "CLAUDE_FAILED no ANTHROPIC_API_KEY and no ~/.config/anthropic/api_key"; exit 1; }
M=$1; TAG=$2; N=$3
mkdir -p results/swarm_$TAG
export SB_USAGE_LOG=results/swarm_$TAG/usage.jsonl
echo "CLAUDE_STARTED $M $TAG $(date -u +%FT%TZ)"
.venv/bin/python -m scb swarm --models $M --tag $TAG --seed 1 --n $N --agents 16 --channels wiki \
  --rules forbidden silent --awareness unaware --listing none --log none --workers 4
echo "CLAUDE_DONE $M $TAG exit $? $(date -u +%FT%TZ)"
