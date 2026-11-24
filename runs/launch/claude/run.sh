#!/bin/bash
# Claude Haiku 5.5 with and without the rule at N = 16 (benchmark entry 22, branch claude-api), run from this machine.
#   bash results/logs/claude/run.sh pilot    2 games per rule, tag claude_pilot (about 770 calls, a few cents)
#   bash results/logs/claude/run.sh full     100 games per rule, tag claude_haiku (38.4k calls)
# The key is read from ANTHROPIC_API_KEY, else from ~/.config/anthropic/api_key (never printed).
set -u
cd "$(dirname "$0")/../../.." || exit 1
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$HOME/.config/anthropic/api_key" ]; then
  ANTHROPIC_API_KEY=$(tr -d '[:space:]' < "$HOME/.config/anthropic/api_key"); export ANTHROPIC_API_KEY
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "CLAUDE_FAILED no ANTHROPIC_API_KEY and no ~/.config/anthropic/api_key"; exit 1; }
case "${1:-}" in
  pilot) TAG=claude_pilot; N=2 ;;
  full)  TAG=claude_haiku; N=100 ;;
  *) echo "usage: run.sh pilot|full"; exit 1 ;;
esac
mkdir -p results/swarm_$TAG
export SB_USAGE_LOG=results/swarm_$TAG/usage.jsonl
echo "CLAUDE_STARTED $TAG $(date -u +%FT%TZ)"
.venv/bin/python -m scb swarm --models claude-haiku-5-5 --tag $TAG --seed 1 --n $N --agents 16 --channels wiki \
  --rules forbidden silent --awareness unaware --listing none --log none --workers 4
echo "CLAUDE_DONE $TAG exit $? $(date -u +%FT%TZ)"
