#!/usr/bin/env bash
# Qwen2.5-7B-Instruct Level 1 hub-OD inference on Deepnet2.
#
# Submit from the LibSignal repository root:
#   sbatch --mcs-label=QSIURP_Salman extras/slurm_qwen25_7b_odh_l1.sh
#
# The model is downloaded into the per-user HF cache before submission; the
# compute job is deliberately offline so it cannot stall on Hub/Xet traffic.

#SBATCH --job-name=qwen25_7b_odh_l1
#SBATCH --output=logs/qwen25_7b_odh_l1_%j.out
#SBATCH --error=logs/qwen25_7b_odh_l1_%j.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_2g.35gb:1
#SBATCH --time=12:00:00
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

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Repository: ${REPO_DIR}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "Python: $(command -v python)"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
PREFIX="${PREFIX:-qwen25_7b_odh_l1}"
echo "PREFIX=${PREFIX}"
python -c "import torch, libsumo, transformers; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}')"

python -u run.py \
  --agent qwen25_7b \
  --world sumo \
  --network sumo4x4 \
  --seed 42 \
  --ngpu 0 \
  --interface libsumo \
  --prefix "${PREFIX}"
