#!/usr/bin/env bash
# Submit realism_full 4x4 batch on gpujobs (login node).
#
# All axes ON: hetero + slow_start + crossing_proxy + obs_penetration 0.8 + gauss sigma 2.
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_realism_full_4x4.sh smoke      # MP only (~12 min)
#   ./extras/submit_realism_full_4x4.sh baselines # MP + FT
#   ./extras/submit_realism_full_4x4.sh rl        # DQN + PressLight + CoLight (~12h each)
#   ./extras/submit_realism_full_4x4.sh all

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
  smoke)
    submit extras/slurm_realism_full_maxpressure.sh
    ;;
  baselines)
    submit extras/slurm_realism_full_maxpressure.sh
    submit extras/slurm_realism_full_fixedtime.sh
    ;;
  rl)
    submit extras/slurm_realism_full_dqn.sh
    submit extras/slurm_realism_full_presslight.sh
    submit extras/slurm_realism_full_colight.sh
    ;;
  all)
    submit extras/slurm_realism_full_maxpressure.sh
    submit extras/slurm_realism_full_fixedtime.sh
    submit extras/slurm_realism_full_dqn.sh
    submit extras/slurm_realism_full_presslight.sh
    submit extras/slurm_realism_full_colight.sh
    ;;
  *)
    echo "Usage: $0 {smoke|baselines|rl|all}"
    exit 1
    ;;
esac

echo "Submitted. Monitor: squeue -u \$USER"
echo "See docs/REALISM_FULL.md for outputs and interpretation."
