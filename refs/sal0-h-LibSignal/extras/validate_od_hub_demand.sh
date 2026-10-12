#!/usr/bin/env bash
# Validate OD hub demand set (stats + native SUMO 1800s smoke).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/bin:${PATH:-}"

python3 extras/gen_od_hub_demand_grid4x4.py --validate-only

NET=data/raw_data/grid4x4/grid4x4.net.xml
for f in fixed_1800 hold_00 train_00; do
  echo "SUMO smoke $f ..."
  sumo -n "$NET" \
    -r "data/raw_data/grid4x4/od_hubs/demand_set/${f}.rou.xml" \
    --end 1800 --no-warnings true --seed 42 >/dev/null
  echo "  OK"
done
echo "All OD hub validation checks passed."
