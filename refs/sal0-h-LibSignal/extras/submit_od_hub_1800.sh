#!/usr/bin/env bash
# Submit / run hub-centric OD 1800s experiments (Level 1 and/or 2).
# Usage:
#   ./extras/submit_od_hub_1800.sh l1          # local sequential baselines + print RL cmds
#   ./extras/submit_od_hub_1800.sh l2
#   ./extras/submit_od_hub_1800.sh all
#   MODE=sbatch MCS_LABEL=crs-XXXX ./extras/submit_od_hub_1800.sh all

set -euo pipefail
cd "$(dirname "$0")/.."
LEVEL="${1:-l1}"
MODE="${MODE:-local}"
SEED="${SEED:-42}"
PREFIX_L1="${PREFIX_L1:-odh_l1}"
PREFIX_L2="${PREFIX_L2:-odh_l2}"

run_one() {
  local agent="$1" prefix="$2"
  echo "=== $agent  prefix=$prefix ==="
  if [[ "$MODE" == "sbatch" ]]; then
    mkdir -p logs
    sbatch --mcs-label="${MCS_LABEL:?set MCS_LABEL}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${agent}
#SBATCH --output=logs/${agent}_%j.out
#SBATCH --error=logs/${agent}_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
cd \$HOME/LibSignalFork
CONDA_PREFIX="\${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"
python run.py -a ${agent} -w sumo -n sumo4x4 --seed ${SEED} --ngpu -1 --interface libsumo --prefix ${prefix}
EOF
  else
    python run.py -a "$agent" -w sumo -n sumo4x4 --seed "$SEED" --ngpu -1 --interface libsumo --prefix "$prefix"
  fi
}

baselines_l1=(maxpressure_odh_l1 fixedtime_odh_l1)
rl_l1=(dqn_odh_l1 presslight_odh_l1 colight_odh_l1)
baselines_l2=(maxpressure_odh_l2 fixedtime_odh_l2)
rl_l2=(dqn_odh_l2 presslight_odh_l2 colight_odh_l2)

run_level() {
  local level="$1"
  if [[ "$level" == "l1" ]]; then
    for a in "${baselines_l1[@]}"; do run_one "$a" "$PREFIX_L1"; done
    if [[ "$MODE" == "sbatch" ]]; then
      for a in "${rl_l1[@]}"; do run_one "$a" "$PREFIX_L1"; done
    else
      echo "RL Level 1 commands (long):"
      for a in "${rl_l1[@]}"; do
        echo "  python run.py -a $a -w sumo -n sumo4x4 --seed $SEED --ngpu -1 --prefix $PREFIX_L1"
      done
    fi
  else
    for a in "${baselines_l2[@]}"; do run_one "$a" "$PREFIX_L2"; done
    if [[ "$MODE" == "sbatch" ]]; then
      for a in "${rl_l2[@]}"; do run_one "$a" "$PREFIX_L2"; done
    else
      echo "RL Level 2 commands (long):"
      for a in "${rl_l2[@]}"; do
        echo "  python run.py -a $a -w sumo -n sumo4x4 --seed $SEED --ngpu -1 --prefix $PREFIX_L2"
      done
    fi
  fi
}

case "$LEVEL" in
  l1) run_level l1 ;;
  l2) run_level l2 ;;
  all) run_level l1; run_level l2 ;;
  *) echo "usage: $0 l1|l2|all"; exit 1 ;;
esac
