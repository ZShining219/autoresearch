#!/usr/bin/env bash
# Submit crossing-proxy 4x4 batch on gpujobs (login node).
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_crossing_proxy_4x4.sh baselines   # FT + MP only (~2h each)
#   ./extras/submit_crossing_proxy_4x4.sh rl           # DQN + PressLight + CoLight (~12h each)
#   ./extras/submit_crossing_proxy_4x4.sh all

set -euo pipefail

cd ~/LibSignalFork
mkdir -p logs

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=crs-XXXX"
  exit 1
fi

MODE="${1:-all}"

submit() {
  local script="$1"
  chmod +x "$script"
  sbatch --mcs-label="${MCS_LABEL}" "$script"
}

case "$MODE" in
  baselines)
    submit extras/slurm_crossing_proxy_fixedtime.sh
    submit extras/slurm_crossing_proxy_maxpressure.sh
    ;;
  rl)
    submit extras/slurm_crossing_proxy_dqn.sh
    submit extras/slurm_crossing_proxy_presslight.sh
    submit extras/slurm_crossing_proxy_colight.sh
    ;;
  all)
    submit extras/slurm_crossing_proxy_fixedtime.sh
    submit extras/slurm_crossing_proxy_maxpressure.sh
    submit extras/slurm_crossing_proxy_dqn.sh
    submit extras/slurm_crossing_proxy_presslight.sh
    submit extras/slurm_crossing_proxy_colight.sh
    ;;
  *)
    echo "Usage: $0 {baselines|rl|all}"
    exit 1
    ;;
esac

echo "Submitted. Monitor: squeue -u \$USER"
