#!/usr/bin/env bash
# Level 1 + Level 2 experiment runner for hub OD 1800s.
# Baselines are short; RL jobs are long — use MODE=sbatch on the cluster.
#
#   ./extras/run_od_hub_experiments.sh level1-baselines
#   ./extras/run_od_hub_experiments.sh level2-baselines
#   MODE=sbatch MCS_LABEL=crs-XXXX ./extras/run_od_hub_experiments.sh level1-all
#   MODE=sbatch MCS_LABEL=crs-XXXX ./extras/run_od_hub_experiments.sh level2-all

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
TARGET="${1:-level1-baselines}"
MODE="${MODE:-local}"
SEED="${SEED:-42}"
# Cluster: libsumo (default). Local macOS Homebrew SUMO: INTERFACE=traci
INTERFACE="${INTERFACE:-libsumo}"

run_agent() {
  local agent="$1" prefix="$2"
  local cmd=(python run.py -a "$agent" -w sumo -n sumo4x4 --seed "$SEED" --ngpu -1 --interface "$INTERFACE" --prefix "$prefix")
  echo ">> ${cmd[*]}"
  if [[ "$MODE" == "sbatch" ]]; then
    sbatch --mcs-label="${MCS_LABEL:?set MCS_LABEL}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${agent}
#SBATCH --output=logs/${agent}_%j.out
#SBATCH --error=logs/${agent}_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
cd "\$HOME/LibSignalFork"
CONDA_PREFIX="\${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"
python run.py -a ${agent} -w sumo -n sumo4x4 --seed ${SEED} --ngpu -1 --interface libsumo --prefix ${prefix}
EOF
  else
    "${cmd[@]}"
  fi
}

case "$TARGET" in
  level1-baselines)
    run_agent maxpressure_odh_l1 odh_l1
    run_agent fixedtime_odh_l1 odh_l1
    ;;
  level1-all)
    run_agent maxpressure_odh_l1 odh_l1
    run_agent fixedtime_odh_l1 odh_l1
    run_agent dqn_odh_l1 odh_l1
    run_agent presslight_odh_l1 odh_l1
    run_agent colight_odh_l1 odh_l1
    ;;
  level2-baselines)
    run_agent maxpressure_odh_l2 odh_l2
    run_agent fixedtime_odh_l2 odh_l2
    ;;
  level2-all)
    run_agent maxpressure_odh_l2 odh_l2
    run_agent fixedtime_odh_l2 odh_l2
    run_agent dqn_odh_l2 odh_l2
    run_agent presslight_odh_l2 odh_l2
    run_agent colight_odh_l2 odh_l2
    ;;
  *)
    echo "usage: $0 level1-baselines|level1-all|level2-baselines|level2-all"
    exit 1
    ;;
esac
