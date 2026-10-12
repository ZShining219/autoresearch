#!/usr/bin/env bash
# Mac only. Rsync L2-axis hetero loggers from gpujobs, check they are not the
# Sep 12 no-op, then print the git commands to push (does not commit for you).
#
#   ./extras/download_and_push_hetero.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

AXES=hetero ./extras/download_l2_axes.sh
echo ""
echo "=== hetero rewrite check ==="
python extras/check_hetero_l2_axis.py
echo ""
echo "If that printed 10/10 cells OK, push with:"
cat <<'EOF'
git add \
  data/output_data/tsc/sumo_fixedtime_odh_l2_hetero \
  data/output_data/tsc/sumo_fixedtime_odh_l2_hetero_1x21 \
  data/output_data/tsc/sumo_maxpressure_odh_l2_hetero \
  data/output_data/tsc/sumo_maxpressure_odh_l2_hetero_1x21 \
  data/output_data/tsc/sumo_dqn_odh_l2_hetero \
  data/output_data/tsc/sumo_dqn_odh_l2_hetero_1x21 \
  data/output_data/tsc/sumo_presslight_odh_l2_hetero \
  data/output_data/tsc/sumo_presslight_odh_l2_hetero_1x21 \
  data/output_data/tsc/sumo_colight_odh_l2_hetero \
  data/output_data/tsc/sumo_colight_odh_l2_hetero_1x21
git commit -m "$(cat <<'MSG'
Add L2 single-axis hetero trip metrics after hub-OD fleet rewrite.

MSG
)"
git push origin HEAD
python paper/ieee/scripts/extract_l2_axis_att.py
EOF
