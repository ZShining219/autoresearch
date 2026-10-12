#!/usr/bin/env bash
# Print crossing_proxy 4x4 ATT summary from LibSignal logs (run on gpujobs or Mac after sync).
#
# Usage:
#   cd ~/LibSignalFork
#   bash extras/summarize_crossing_proxy_results.sh

set -euo pipefail

cd "${1:-$HOME/LibSignalFork}"
ROOT="data/output_data/tsc"

echo "=== Baselines (BRF Final Travel Time) ==="
grep -H "Final Travel Time" \
  "${ROOT}/sumo_maxpressure_crossing_proxy/sumo4x4/baseline_crossing_proxy/logger/"*_BRF.log \
  "${ROOT}/sumo_fixedtime_crossing_proxy/sumo4x4/baseline_crossing_proxy/logger/"*_BRF.log \
  2>/dev/null || true

echo ""
echo "=== RL final test ATT (last TEST row in DTL) ==="
for agent in dqn presslight colight; do
  dir="${ROOT}/sumo_${agent}_crossing_proxy/sumo4x4/crossing_proxy_4x4/logger"
  dtl=$(ls -1 "${dir}/"*_DTL.log 2>/dev/null | tail -1 || true)
  if [[ -z "${dtl}" ]]; then
    echo "${agent}: (no DTL log)"
    continue
  fi
  last=$(grep $'\tTEST\t' "${dtl}" | tail -1)
  ep=$(echo "${last}" | cut -f3)
  att=$(echo "${last}" | cut -f4)
  echo "${agent}: episode=${ep}  final_test_att=${att}s  (${dtl##*/})"
done

echo ""
echo "=== RL best test ATT (min TEST travel_time) ==="
python3 - <<'PY'
import re
from pathlib import Path

cols = ["model", "split", "episode", "travel_time", "c5", "reward", "queue", "delay", "throughput"]
root = Path("data/output_data/tsc")
for name in ["dqn", "presslight", "colight"]:
    d = root / f"sumo_{name}_crossing_proxy/sumo4x4/crossing_proxy_4x4/logger"
    logs = sorted(d.glob("*_DTL.log"))
    if not logs:
        print(f"{name}: no DTL")
        continue
    import pandas as pd
    best_att, best_log, final_att = None, None, None
    for p in logs:
        df = pd.read_csv(p, sep="\t", header=None, names=cols)
        te = df[df.split == "TEST"]
        if te.empty:
            continue
        att_min = te.travel_time.min()
        att_final = te.travel_time.iloc[-1]
        if best_att is None or att_min < best_att:
            best_att, best_log = att_min, p.name
        final_att = att_final
    print(f"{name}: best_test={best_att:.2f}s  final_test={final_att:.2f}s  log={best_log}")
PY
