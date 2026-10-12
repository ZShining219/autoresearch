#!/usr/bin/env bash
# Run default sumo4x4 new_metrics exports with nohup (no SLURM).
# Use on Salman server / any machine without sbatch.
#
#   cd ~/LibSignalFork
#   conda activate traffic   # or libsignal
#   export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
#   export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"
#   ./extras/run_new_metrics_4x4.sh
#
# Or one agent only:
#   AGENT=maxpressure ./extras/run_new_metrics_4x4.sh
#
# Monitor:
#   tail -f logs/new_metrics_maxpressure_seed42_steps3600.log
#
# Outputs: extras/output/<agent>/sumo4x4/seed42_steps3600/

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/LibSignalFork}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
TEST_STEPS="${TEST_STEPS:-3600}"
RUN_NAME="${RUN_NAME:-seed42_steps3600}"
DQN_EPISODES="${DQN_EPISODES:-200}"
PYTHON="${PYTHON:-python}"

export PYTHONHASHSEED=0

cd "${REPO_DIR}"
mkdir -p logs

run_one() {
  local agent="$1"
  local train_flag="${2:-}"
  local log_file="${REPO_DIR}/logs/new_metrics_${agent}_${RUN_NAME}.log"
  local -a extra_args=()
  if [[ "${train_flag}" == "train" ]]; then
    extra_args+=(--train --episodes "${DQN_EPISODES}")
  fi
  echo "Starting ${agent} -> ${log_file}"
  nohup "${PYTHON}" extras/run_new_metrics.py \
    --agent "${agent}" \
    --network "${NETWORK}" \
    --seed "${SEED}" \
    --test-steps "${TEST_STEPS}" \
    --run-name "${RUN_NAME}" \
    --prefix new_metrics \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1 &
  echo "  PID $!"
}

if [[ -n "${AGENT:-}" ]]; then
  case "${AGENT}" in
    dqn) run_one dqn train ;;
    *)   run_one "${AGENT}" ;;
  esac
  exit 0
fi

# Default: all three (MP and FT in parallel; DQN long — also background)
run_one maxpressure
run_one fixedtime
run_one dqn train

echo ""
echo "All started in background. Monitor:"
echo "  tail -f logs/new_metrics_maxpressure_${RUN_NAME}.log"
echo "  tail -f logs/new_metrics_fixedtime_${RUN_NAME}.log"
echo "  tail -f logs/new_metrics_dqn_${RUN_NAME}.log"
echo ""
echo "Outputs when done:"
echo "  extras/output/{maxpressure,fixedtime,dqn}/${NETWORK}/${RUN_NAME}/new_metrics.csv"
