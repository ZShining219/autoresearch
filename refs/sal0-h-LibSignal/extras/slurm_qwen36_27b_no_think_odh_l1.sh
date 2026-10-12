#!/usr/bin/env bash
# Qwen3.6-27B no-thinking Level-1 hub-OD inference on a full H200 7g slice.
# Submit with:
#   sbatch --mcs-label=QSIURP_Salman extras/slurm_qwen36_27b_no_think_odh_l1.sh

#SBATCH --job-name=qwen36_27b_no_think
#SBATCH --output=logs/qwen36_27b_no_think_%j.out
#SBATCH --error=logs/qwen36_27b_no_think_%j.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_7g.141gb:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
# The Qwen3.6 snapshot was prefetched with hf --cache-dir at this root.
export HF_HOME=/data1/shajizad/.cache/huggingface
export HF_HUB_CACHE=/data1/shajizad/.cache/huggingface

AGENT="qwen36_27b_no_think"
PREFIX="${PREFIX:-qwen36_27b_no_think_odh_l1}"
echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Repository: ${REPO_DIR}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"
echo "Python: $(command -v python)"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "AGENT=${AGENT} PREFIX=${PREFIX}"
echo "HF_HOME=${HF_HOME} HF_HUB_CACHE=${HF_HUB_CACHE}"
python -c "import torch, libsumo, transformers; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}')"

python -u run.py \
  --agent "${AGENT}" \
  --world sumo \
  --network sumo4x4 \
  --seed 42 \
  --ngpu 0 \
  --interface libsumo \
  --prefix "${PREFIX}"
