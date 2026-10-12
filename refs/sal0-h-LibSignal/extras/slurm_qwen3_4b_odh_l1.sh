#!/usr/bin/env bash
# Qwen3-4B Level 1 hub-OD inference on the 1g.18gb MIG slice.
# The AGENT variable selects one of the chained Qwen3 configs:
# qwen3_4b_no_think, qwen3_4b_think1024, or qwen3_4b_think2048.

#SBATCH --job-name=qwen3_4b_odh_l1
#SBATCH --output=logs/qwen3_4b_odh_l1_%j.out
#SBATCH --error=logs/qwen3_4b_odh_l1_%j.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_DIR}"
CONDA_PREFIX="${CONDA_PREFIX:-/data1/shajizad/.conda/envs/myenv}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export SUMO_HOME="${SUMO_HOME:-$("${CONDA_PREFIX}/bin/python" -c 'import os, sumo; print(os.path.dirname(sumo.__file__))')}"
export PATH="${SUMO_HOME}/bin:${PATH}"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

AGENT="${AGENT:-qwen3_4b_no_think}"
PREFIX="${PREFIX:-${AGENT}_odh_l1}"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Repository: ${REPO_DIR}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "Python: $(command -v python)"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "AGENT=${AGENT} PREFIX=${PREFIX}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
python -c "import torch, libsumo, transformers; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}')"

python -u run.py \
  --agent "${AGENT}" \
  --world sumo \
  --network sumo4x4 \
  --seed 42 \
  --ngpu 0 \
  --interface libsumo \
  --prefix "${PREFIX}"
