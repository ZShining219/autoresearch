#!/usr/bin/env bash
# Local Traffic-R1 Level 1 hub-OD inference on Deepnet2.
#
# Submit from the LibSignal repository root after replacing the MCS label with
# the label assigned to your project:
#
#   mkdir -p logs
#   sbatch --export=ALL --mcs-label=QSIURP_Salman \
#     extras/slurm_traffic_r1_odh_l1.sh
#
# The model is the local Season998/Traffic-R1 checkpoint via in-process vLLM
# (configs/tsc/traffic_r1.yml backend=vllm). Install vllm in $CONDA_ENV against
# the same torch/CUDA as the job. It does not use OPENAI_API_KEY. Inference-only
# over the three held-out Level 1 OD route files in od_hub_1800_base.yml.

#SBATCH --job-name=traffic_r1_odh_l1
#SBATCH --output=logs/traffic_r1_odh_l1_%j.out
#SBATCH --error=logs/traffic_r1_odh_l1_%j.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-traffic}"
cd "${REPO_DIR}"

# Keep this usable both when sbatch exports an already-active environment and
# when the batch shell has to activate it itself.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in the batch environment; submit with --export=ALL or load conda first" >&2
    exit 1
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# setup.sh normally persists SUMO_HOME in the conda environment. Derive it
# only as a fallback for environments where that activation hook is missing.
if [[ -z "${SUMO_HOME:-}" ]]; then
  if SUMO_HOME_FROM_PYTHON="$(python -c 'import os, sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null)"; then
    export SUMO_HOME="${SUMO_HOME_FROM_PYTHON}"
  elif [[ -d "${CONDA_PREFIX}/share/sumo" ]]; then
    export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
  else
    echo "SUMO_HOME is unset and could not be derived from the active environment" >&2
    exit 1
  fi
fi
export PATH="${SUMO_HOME}/bin:${PATH}"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Repository: ${REPO_DIR}"
echo "Python: $(command -v python)"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python run.py \
  --agent traffic_r1 \
  --world sumo \
  --network sumo4x4 \
  --seed 42 \
  --ngpu 0 \
  --interface libsumo \
  --prefix odh_l1