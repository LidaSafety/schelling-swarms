#!/bin/bash
# Submit one Slurm job per model in configs/models.yaml, with as many GPUs as the model's tensor-parallel size.
#   CONDA_ENV=myenv bash scripts/submit_slurm.sh [extra sbatch args, e.g. --partition=gpu]
#   ONLY="deaddrop_drive committee_sealed" CONDA_ENV=myenv bash scripts/submit_slurm.sh   # a subset of experiments
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/slurm
mapfile -t ROWS < <(python -c "
import yaml
for m in yaml.safe_load(open('configs/models.yaml'))['models']:
    print(m['name'], m.get('tp', 1))
")
for row in "${ROWS[@]}"; do
  name=${row% *}; tp=${row#* }
  jid=$(sbatch --parsable "$@" --gres=gpu:"$tp" --job-name="scb_$name" \
        --export=ALL,MODEL="$name",ONLY="${ONLY:-}" scripts/job.sbatch)
  echo "submitted $name (gpus=$tp) as job $jid"
done
