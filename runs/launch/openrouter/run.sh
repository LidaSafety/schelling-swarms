#!/bin/bash
# One OpenRouter model with and without the rule at N = 16 (entry 23 of experiments.yaml), run from this machine.
#   bash results/logs/openrouter/run.sh <slug> <tag> <games>
#   e.g. bash results/logs/openrouter/run.sh openai/gpt-oss-120b or_pilot 2
# The model's settings come from configs/models_openrouter.yaml. The key is read from OPENROUTER_API_KEY, else from
# ~/.config/openrouter/api_key (never printed). Each model gets its own folder: results/swarm_<tag>_<slug with _>.
set -u
cd "$(dirname "$0")/../../.." || exit 1
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/.config/openrouter/api_key" ]; then
  OPENROUTER_API_KEY=$(tr -d '[:space:]' < "$HOME/.config/openrouter/api_key"); export OPENROUTER_API_KEY
fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OR_FAILED no OPENROUTER_API_KEY and no ~/.config/openrouter/api_key"; exit 1; }
M=$1; TAG=$2; N=$3
SAFE=$(echo "$M" | tr '/.:' '___')
export WORKERS_FILE=$(mktemp)
SB_ENDPOINTS=$(.venv/bin/python - "$M" <<'EOF'
import json, os, sys, yaml
m = next((x for x in yaml.safe_load(open("configs/models_openrouter.yaml"))["models"] if x["name"] == sys.argv[1]), None)
if m is None:
    sys.exit(f"{sys.argv[1]} is not in configs/models_openrouter.yaml")
body = {"provider": {"require_parameters": True}, "usage": {"include": True}}
if "host" in m:
    body["provider"].update(order=[m["host"]], allow_fallbacks=False)
if "reasoning" in m:
    body["reasoning"] = m["reasoning"]
ep = {"base_url": "https://openrouter.ai/api/v1", "api_key": os.environ["OPENROUTER_API_KEY"], "extra_body": body,
      "token_mult": m.get("token_mult", 1), "retries": m.get("retries", 3)}
print(json.dumps({m["name"]: ep}))
open(os.environ["WORKERS_FILE"], "w").write(str(m.get("workers", 4)))
EOF
) || { echo "OR_FAILED $M settings"; exit 1; }
WORKERS=$(cat "$WORKERS_FILE"); rm -f "$WORKERS_FILE"
export SB_ENDPOINTS
mkdir -p results/swarm_${TAG}_$SAFE
export SB_USAGE_LOG=results/swarm_${TAG}_$SAFE/usage.jsonl
echo "OR_STARTED $M $TAG $(date -u +%FT%TZ)"
.venv/bin/python -m scb swarm --models "$M" --tag ${TAG}_$SAFE --seed 1 --n $N --agents 16 --channels wiki \
  --rules forbidden silent --awareness unaware --listing none --log none --workers $WORKERS
echo "OR_DONE $M $TAG exit $? $(date -u +%FT%TZ)"
