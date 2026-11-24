#!/bin/bash
# The capability quiz (entry 25) for the API models, from this machine, with each model's game settings.
#   bash results/logs/capability_api.sh <model>     claude-* through the Anthropic API, others through OpenRouter
set -u
cd "$(dirname "$0")/../.." || exit 1
M=$1; W=8
if [[ $M == claude-* ]]; then
  [ -n "${ANTHROPIC_API_KEY:-}" ] || { ANTHROPIC_API_KEY=$(tr -d '[:space:]' < "$HOME/.config/anthropic/api_key"); export ANTHROPIC_API_KEY; }
else
  [ -n "${OPENROUTER_API_KEY:-}" ] || { OPENROUTER_API_KEY=$(tr -d '[:space:]' < "$HOME/.config/openrouter/api_key"); export OPENROUTER_API_KEY; }
  SB_ENDPOINTS=$(.venv/bin/python - "$M" <<'PY'
import json, os, sys, yaml
m = next(x for x in yaml.safe_load(open("configs/models_openrouter.yaml"))["models"] if x["name"] == sys.argv[1])
body = {"provider": {"require_parameters": True}, "usage": {"include": True}}
if "host" in m: body["provider"].update(order=[m["host"]], allow_fallbacks=False)
if "reasoning" in m: body["reasoning"] = m["reasoning"]
print(json.dumps({m["name"]: {"base_url": "https://openrouter.ai/api/v1", "api_key": os.environ["OPENROUTER_API_KEY"],
                              "extra_body": body, "token_mult": m.get("token_mult", 1), "retries": 8}}))
PY
) || exit 1
  export SB_ENDPOINTS; W=4
fi
echo "CAP_STARTED $M $(date -u +%FT%TZ)"
.venv/bin/python -m scb capability --models "$M" --subset results/capability_data/mmlu_pro_504.jsonl --tag mmlupro \
  --seed 1 --workers $W
echo "CAP_DONE $M exit $? $(date -u +%FT%TZ)"
