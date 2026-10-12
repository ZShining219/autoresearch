#!/usr/bin/env bash
# Pull L2 single-axis loggers from gpujobs onto this machine.
#
# Usage (Mac, repo root):
#   ./extras/download_l2_axes.sh
#   AXES="hetero noise obs crossing_proxy" ./extras/download_l2_axes.sh
#
# Slow-start is omitted by default (those jobs failed). After a successful
# resubmit, run: AXES=slow_start ./extras/download_l2_axes.sh

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
REMOTE_TSC="${REMOTE_TSC:-~/LibSignalFork/data/output_data/tsc}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_TSC="${REPO_ROOT}/data/output_data/tsc"
AXES="${AXES:-hetero crossing_proxy obs noise}"

mkdir -p "${LOCAL_TSC}"

echo "Listing remote L2 axis dirs on ${SERVER}..."
# shellcheck disable=SC2086
REMOTE_DIRS="$(ssh "${SERVER}" "bash -lc '
  cd ${REMOTE_TSC} 2>/dev/null || exit 0
  for ax in ${AXES}; do
    ls -d sumo_*_odh_l2_\${ax} sumo_*_odh_l2_\${ax}_1x21 2>/dev/null || true
  done
'")"

if [[ -z "${REMOTE_DIRS}" ]]; then
  echo "No matching remote dirs for axes: ${AXES}"
  exit 1
fi

echo "${REMOTE_DIRS}"
echo ""

while IFS= read -r name; do
  [[ -z "${name}" ]] && continue
  echo "rsync logger/  ${name}"
  mkdir -p "${LOCAL_TSC}/${name}"
  rsync -az \
    --include='*/' \
    --include='**/logger/***' \
    --exclude='replay/***' \
    --exclude='model/***' \
    --exclude='*' \
    "${SERVER}:${REMOTE_TSC}/${name}/" \
    "${LOCAL_TSC}/${name}/"
done <<< "${REMOTE_DIRS}"

echo ""
echo "Local cells with new_metrics:"
find "${LOCAL_TSC}" -path '*/l2_axis_*/logger/new_metrics*.csv' \
  | sed 's|.*/sumo_\([^/]*\)/\([^/]*\)/l2_axis_\([^/]*\)/.*|\2  \3  \1|' \
  | sort -u
