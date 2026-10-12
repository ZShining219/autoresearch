#!/usr/bin/env bash
# Stage crossing_proxy 4x4 experiment logs for git (run on gpujobs after experiments finish).
#
# Keeps: canonical MP baseline (178.42), FT baseline, RL DTL+BRF (200 ep).
# Skips: crossing_proxy_smoke debug runs and duplicate identical BRF files.
#
# Usage:
#   cd ~/LibSignalFork
#   git pull origin feat/crossing-proxy
#   bash extras/commit_crossing_proxy_logs.sh
#   git status
#   git commit -m "Add crossing_proxy 4x4 experiment logs (seed 42)."

set -euo pipefail

cd "${1:-$HOME/LibSignalFork}"

MP_SMOKE="data/output_data/tsc/sumo_maxpressure_crossing_proxy/sumo4x4/crossing_proxy_smoke/logger/2026_07_17-23_39_53_BRF.log"
MP_BASE="data/output_data/tsc/sumo_maxpressure_crossing_proxy/sumo4x4/baseline_crossing_proxy/logger"

# Notebook expects baseline_crossing_proxy; promote the fixed MP run if SLURM used smoke only.
if [[ ! -f "${MP_BASE}/2026_07_17-23_39_53_BRF.log" && -f "${MP_SMOKE}" ]]; then
  mkdir -p "${MP_BASE}"
  cp "${MP_SMOKE}" "${MP_BASE}/"
  echo "Copied canonical MP baseline (178.42 s) -> ${MP_BASE}/"
fi

git add \
  "${MP_BASE}/"*.log \
  data/output_data/tsc/sumo_fixedtime_crossing_proxy/sumo4x4/baseline_crossing_proxy/logger/2026_07_17-23_43_13_BRF.log \
  data/output_data/tsc/sumo_dqn_crossing_proxy/sumo4x4/crossing_proxy_4x4/logger/*.log \
  data/output_data/tsc/sumo_presslight_crossing_proxy/sumo4x4/crossing_proxy_4x4/logger/*.log \
  data/output_data/tsc/sumo_colight_crossing_proxy/sumo4x4/crossing_proxy_4x4/logger/*.log

echo ""
echo "Staged crossing_proxy logs. Skipped: crossing_proxy_smoke (debug), duplicate FT BRF."
echo "Verify: git status"
