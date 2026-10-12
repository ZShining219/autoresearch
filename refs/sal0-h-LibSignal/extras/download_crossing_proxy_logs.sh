#!/usr/bin/env bash
# Download crossing_proxy 4x4 experiment logs from gpujobs to local data/output_data/tsc/.
#
# Usage (from repo root on your Mac):
#   chmod +x extras/download_crossing_proxy_logs.sh
#   ./extras/download_crossing_proxy_logs.sh

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="~/LibSignalFork/data/output_data/tsc"
LOCAL="${REPO_ROOT}/data/output_data/tsc"

mkdir -p "${LOCAL}"

for agent in \
  sumo_maxpressure_crossing_proxy \
  sumo_fixedtime_crossing_proxy \
  sumo_dqn_crossing_proxy \
  sumo_presslight_crossing_proxy \
  sumo_colight_crossing_proxy
do
  echo "Downloading ${agent}..."
  scp -r "${SERVER}:${REMOTE}/${agent}" "${LOCAL}/" || echo "  (skip — not found on server)"
done

echo "Downloading SLURM captures (ped-call stdout)..."
mkdir -p "${REPO_ROOT}/logs"
scp "${SERVER}:~/LibSignalFork/logs/"*crossing* "${REPO_ROOT}/logs/" 2>/dev/null || echo "  (skip — no logs/*crossing* on server)"

echo ""
echo "Done. Summarize on Mac:"
echo "  grep -H 'Final Travel Time' data/output_data/tsc/sumo_*crossing_proxy/sumo4x4/*/logger/*BRF.log"
echo "  grep 'TEST' data/output_data/tsc/sumo_*crossing_proxy/sumo4x4/crossing_proxy_4x4/logger/*DTL.log | tail -5"
