#!/usr/bin/env bash
# gpujobs only. Pull the CoLight remap, then submit the 18 cells that still
# need it. Skips L2-axis hetero (already resubmitted after 805dc4b).
#
#   ssh mmirzata@172.20.48.59
#   export MCS_LABEL=15288
#   bash ~/LibSignalFork/extras/server_submit_colight_graph_fix.sh
#
# First pull if this file is not on the server yet: paste the body below
# after `cd ~/LibSignalFork && git pull --no-rebase`.

set -euo pipefail

cd "${HOME}/LibSignalFork"

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288" >&2
  exit 1
fi

echo "=== git pull ==="
git pull --no-rebase
echo "HEAD: $(git log -1 --oneline)"

if [[ ! -f common/colight_graph.py ]]; then
  echo "ERROR: common/colight_graph.py missing. Pull did not get 2eff2d7+." >&2
  exit 1
fi
grep -q remap_graph_edges_to_world agent/colight.py \
  || { echo "ERROR: CoLight remap not in agent/colight.py" >&2; exit 1; }
grep -q rewrite_route_file_hetero_mix world/world_sumo.py \
  || { echo "ERROR: hetero rewrite not in world/world_sumo.py" >&2; exit 1; }

echo ""
echo "=== jobs that will be submitted (18; hetero axis skipped) ==="
./extras/submit_colight_table_rerun.sh list graph-fix
echo ""
./extras/submit_colight_table_rerun.sh graph-fix
echo ""
echo "Monitor: squeue -u \$USER -o '%.8i %.40j %.8T %.10M %R'"
echo "A live job must print: [CoLight] remapped N/M graph edges into world intersection order"
