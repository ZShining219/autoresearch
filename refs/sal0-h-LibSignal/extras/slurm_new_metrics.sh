#!/usr/bin/env bash
# Submit from gpujobs LOGIN node (do NOT run 7x28 directly on login):
#
#   cd ~/LibSignalFork
#   mkdir -p logs
#   export MCS_LABEL=crs-XXXX          # your course label
#   export CONDA_ENV=libsignal         # must have libsumo
#   export NETWORK=sumo4x4             # default 4x4 baseline study
#   export TEST_STEPS=3600
#   export RUN_NAME=seed42_steps3600
#   export TRAIN=1                     # for dqn only
#   export DQN_EPISODES=200
#   sbatch --mcs-label="${MCS_LABEL}" extras/slurm_new_metrics.sh
#
# Or submit all three (MP, FT, DQN):
#   ./extras/submit_new_metrics_4x4.sh
#
# Note: gpujobs has no "deepnet" partition. Jobs are routed via --mcs-label only.
# List partitions if needed: sinfo
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/new_metrics_<jobid>.out
#
# Outputs: extras/output/<agent>/<network>/<run_name>/
#   new_metrics.csv
#   new_metrics_meta.json

#SBATCH --job-name=new_metrics
#SBATCH --output=logs/new_metrics_%j.out
#SBATCH --error=logs/new_metrics_%j.err
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-libsignal}"
REPO_DIR="${REPO_DIR:-$HOME/LibSignalFork}"
AGENT="${AGENT:-maxpressure}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
TEST_STEPS="${TEST_STEPS:-3600}"
RUN_NAME="${RUN_NAME:-seed42_steps3600}"
INTERFACE="${INTERFACE:-libsumo}"
TRAIN="${TRAIN:-0}"
DQN_EPISODES="${DQN_EPISODES:-200}"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

export SUMO_HOME="${SUMO_HOME:-${CONDA_PREFIX}/share/sumo}"
export PATH="${SUMO_HOME}/bin:${PATH}"

RUN_ARGS=(
  --agent "${AGENT}"
  --network "${NETWORK}"
  --seed "${SEED}"
  --test-steps "${TEST_STEPS}"
  --interface "${INTERFACE}"
  --run-name "${RUN_NAME}"
  --prefix new_metrics
)

if [[ "${TRAIN}" == "1" ]]; then
  RUN_ARGS+=(--train --episodes "${DQN_EPISODES}")
fi

echo "Host:      $(hostname)"
echo "Repo:      ${REPO_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Network:   ${NETWORK}"
echo "Run name:  ${RUN_NAME}"
python -c "import libsumo; print('libsumo: OK')" 2>/dev/null || {
  echo "WARNING: libsumo not found in ${CONDA_ENV} on $(hostname)."
  echo "Try: export CONDA_ENV=libsignal  (or libsignal/traffic on login: python -c 'import libsumo')"
  echo "Continuing — run_new_metrics.py will fall back to traci (much slower on 7x28)."
}

python extras/run_new_metrics.py "${RUN_ARGS[@]}"

OUT_DIR="${REPO_DIR}/extras/output/${AGENT}/${NETWORK}/${RUN_NAME}"
echo ""
echo "Done. new_metrics:"
head -1 "${OUT_DIR}/new_metrics.csv"
echo ""
echo "CSV:  ${OUT_DIR}/new_metrics.csv"
echo "Meta: ${OUT_DIR}/new_metrics_meta.json"
