#!/usr/bin/env bash
# Generate Ingolstadt (sumo1x21) realism + OD-hub assets needed before submitting jobs.
#
# Run once locally (or on the server after git pull):
#   export SUMO_HOME=...   # or use conda libsignal env
#   ./extras/prepare_ingolstadt_1x21_assets.sh
#
# Produces:
#   data/raw_data/ingolstadt21/ingolstadt21_hetero.rou.xml
#   data/raw_data/ingolstadt21/ingolstadt21_slow_start.rou.xml
#   data/raw_data/ingolstadt21/crossing_proxy_lanes.json
#   data/raw_data/ingolstadt21/od_hubs/taz.xml
#   data/raw_data/ingolstadt21/od_hubs/demand_set/{fixed,train_*,hold_*}.rou.xml

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SUMO_HOME:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
  fi
fi
if [[ -z "${SUMO_HOME:-}" ]]; then
  echo "Set SUMO_HOME (needed for od2trips/duarouter and crossing_proxy gen)."
  exit 1
fi
export SUMO_HOME
export PATH="${SUMO_HOME}/bin:${PATH}"
echo "SUMO_HOME=${SUMO_HOME}"

python extras/gen_hetero_routes.py
python extras/gen_slow_start_routes.py
python extras/gen_crossing_proxy_lanes.py --network ingolstadt21
python extras/gen_ingolstadt21_taz.py
python extras/gen_od_hub_demand_ingolstadt21.py

echo "Assets ready. Next: push branch, rsync demand_set if needed, then submit_ingolstadt_1x21_matrix.sh"
