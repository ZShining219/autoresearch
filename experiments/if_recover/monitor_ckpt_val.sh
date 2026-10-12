#!/bin/bash
# Watch selfcond ckpt; on each new save, closed-loop eval WITHOUT autoreg on
# held-out validation flows (var044/var046 x {W, W,E}). Mean ATT = selection
# metric. Test flows (anon_4_4_hangzhou_real.json / _5816.json) are NOT used
# for model selection.
#
# Usage: monitor_ckpt_val.sh [log] [train_pid] [ckpt]
cd /data/users/zfh/workspace/projects/auto_research
PY=/data/users/zfh/workspace/projects/tarl_reproduction/py310/bin/python
LOG="${1:-data/if_recover/sc_eval_track_val.log}"
TRAIN_PID="${2:-}"
CKPT="${3:-results/if_recover/model_if_sc.pt}"
BEST_CKPT="results/if_recover/model_if_sc_best_valatt.pt"
FLOWS_DIR=data/if_recover/flows
TMP=/tmp/ckpt_eval_val_$$.pt

# single instance
exec 9>/tmp/if_monitor_val.lock
flock -n 9 || { echo "another val monitor already running"; exit 1; }

if [ -z "$TRAIN_PID" ]; then
  TRAIN_PID=$(pgrep -f "train_if.py.*selfcond" | head -1)
fi
echo "monitor start: ckpt=$CKPT train_pid=$TRAIN_PID log=$LOG" >> "$LOG"

BEST_ATT=99999
LAST=""
while true; do
  if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "training pid $TRAIN_PID gone, monitor exit" >> "$LOG"
    break
  fi
  T=$(stat -c %Y "$CKPT" 2>/dev/null)
  if [ -n "$T" ] && [ "$T" != "$LAST" ]; then
    sleep 5
    cp "$CKPT" "$TMP"
    ATTS=""
    for spec in "var044 W" "var044 W,E" "var046 W" "var046 W,E"; do
      read -r V D <<< "$spec"
      WD="data/if_recover/rundirs/val_${V}_${D//,/}_$$"
      R=$($PY experiments/if_recover/eval_closed.py \
          --flow "$FLOWS_DIR/anon_4_4_hangzhou_real_5816_${V}.json" \
          --arm if --dirs "$D" --ckpt "$TMP" \
          --workdir "$WD" 2>/dev/null | tail -1)
      A=$(echo "$R" | $PY -c "import json,sys
try: print(json.loads(sys.stdin.read())['att'])
except Exception: print('')" 2>/dev/null)
      ATTS="$ATTS${ATTS:+,}$A"
    done
    EP=$(grep -c "^{.ep." data/if_recover/train_sc.log)
    MEAN=$($PY -c "import sys
v=[float(x) for x in '$ATTS'.split(',') if x]
print('%.2f' % (sum(v)/len(v)) if v else 'nan')" 2>/dev/null)
    echo "ep=$EP ts=$T val_mean_att=$MEAN atts=[$ATTS]" >> "$LOG"
    if [ "$MEAN" != "nan" ] && [ -n "$MEAN" ] && \
       $PY -c "import sys; sys.exit(0 if $MEAN < $BEST_ATT else 1)" 2>/dev/null; then
      BEST_ATT=$MEAN
      cp "$TMP" "${BEST_CKPT}.tmp" && mv "${BEST_CKPT}.tmp" "$BEST_CKPT"
      echo "  ^^ new best val_mean_att=$BEST_ATT -> $BEST_CKPT" >> "$LOG"
    fi
    LAST=$T
  fi
  sleep 30
done
rm -f "$TMP"
