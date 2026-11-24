#!/bin/bash
# Stop this pod from inside it: runpodctl with the pod's own RUNPOD_API_KEY and RUNPOD_POD_ID (from /proc/1/environ,
# which an ssh session does not inherit). Only after the results are pulled: /workspace is the container disk here.
set -u
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | xargs)
echo "STOPPING $RUNPOD_POD_ID $(date -u +%FT%TZ)" >> /workspace/logs/stop.log
runpodctl stop pod "$RUNPOD_POD_ID" >> /workspace/logs/stop.log 2>&1
