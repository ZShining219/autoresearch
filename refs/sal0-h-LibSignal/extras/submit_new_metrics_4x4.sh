#!/usr/bin/env bash
# Submit default sumo4x4 new_metrics exports (MaxPressure, FixedTime, DQN) on gpujobs.
#
# Login node (set your MCS label):
#   cd ~/LibSignalFork
#   git pull
#   mkdir -p logs
#   export MCS_LABEL=crs-XXXX
#   export CONDA_ENV=libsignal
#   ./extras/submit_new_metrics_4x4.sh
#
# Outputs: extras/output/<agent>/sumo4x4/seed42_steps3600/
#   new_metrics.csv
#   new_metrics_meta.json
#
# Monitor: squeue -u $USER

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/LibSignalFork}"
MCS_LABEL="${MCS_LABEL:?Set MCS_LABEL (e.g. export MCS_LABEL=crs-XXXX)}"
CONDA_ENV="${CONDA_ENV:-libsignal}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
TEST_STEPS="${TEST_STEPS:-3600}"
RUN_NAME="${RUN_NAME:-seed42_steps3600}"
DQN_EPISODES="${DQN_EPISODES:-200}"

cd "${REPO_DIR}"
mkdir -p logs

submit_one() {
  local agent="$1"
  local time_limit="$2"
  local extra_env="${3:-}"
  echo "Submitting ${agent} (${time_limit})..."
  sbatch --mcs-label="${MCS_LABEL}" \
    --job-name="new_metrics_${agent}" \
    --output="logs/new_metrics_${agent}_%j.out" \
    --error="logs/new_metrics_${agent}_%j.err" \
    --time="${time_limit}" \
    --cpus-per-task=8 \
    --mem=32G \
    --nodes=1 \
    --ntasks=1 \
    --export=ALL,REPO_DIR="${REPO_DIR}",CONDA_ENV="${CONDA_ENV}",AGENT="${agent}",NETWORK="${NETWORK}",SEED="${SEED}",TEST_STEPS="${TEST_STEPS}",RUN_NAME="${RUN_NAME}"${extra_env} \
    extras/slurm_new_metrics.sh
}

# Baselines: ~minutes each
submit_one maxpressure "01:00:00"
submit_one fixedtime "01:00:00"

# DQN: train 200 ep then export (~hours on CPU)
submit_one dqn "24:00:00" ",TRAIN=1,DQN_EPISODES=${DQN_EPISODES}"

echo ""
echo "Submitted 3 jobs. When done, pull CSVs from:"
echo "  ${REPO_DIR}/extras/output/{maxpressure,fixedtime,dqn}/${NETWORK}/${RUN_NAME}/"
