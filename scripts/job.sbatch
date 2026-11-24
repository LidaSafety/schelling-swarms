#!/bin/bash
# One Slurm GPU job: serve one model with vLLM and run its experiments (scripts/sweep.py).
#   sbatch --gres=gpu:2 --export=ALL,MODEL=gemma-3-27b scripts/job.sbatch
# Or submit every model at once: bash scripts/submit_slurm.sh
# Adjust partition, time and memory to your cluster (or pass them on the sbatch command line).
#SBATCH --job-name=scb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Python environment with vllm, openai and pyyaml: a conda env (CONDA_ENV) or a virtualenv (VENV).
if [ -n "${CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -n "${VENV:-}" ]; then
  source "$VENV/bin/activate"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "node: $(hostname)  gpus: ${CUDA_VISIBLE_DEVICES:-none}"
ARGS=(--models "${MODEL:?set MODEL=<model name from configs/models.yaml>}")
[ -n "${ONLY:-}" ] && ARGS+=(--only ${ONLY//[,+]/ })
python scripts/sweep.py "${ARGS[@]}"
