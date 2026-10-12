#!/bin/bash
# Watch model_if_sc.pt; on each new save, closed-loop eval on HangZhou2 (5816).
# Archives the best-ATT checkpoint separately (val-best != control-best).
cd /data/users/zfh/workspace/projects/auto_research
FLOW="${2:-refs/RobustLight/data/Hangzhou/4_4/anon_4_4_hangzhou_real_5816.json}"
LOG="${1:-data/if_recover/sc_eval_track.log}"
BEST_CKPT="results/if_recover/model_if_sc_best_att.pt"
BEST_ATT=99999
LAST=""
while kill -0 12622 2>/dev/null; do
  T=$(stat -c %Y results/if_recover/model_if_sc.pt 2>/dev/null)
  if [ -n "$T" ] && [ "$T" != "$LAST" ]; then
    sleep 5
    cp results/if_recover/model_if_sc.pt /tmp/ckpt_eval_5816.pt
    R=$(/data/users/zfh/workspace/projects/tarl_reproduction/py310/bin/python \
        experiments/if_recover/eval_closed.py --flow "$FLOW" --arm if --dirs W \
        --ckpt /tmp/ckpt_eval_5816.pt --autoreg \
        --workdir data/if_recover/rundirs/if_sc_auto_5816 2>/dev/null | tail -1)
    EP=$(grep -c "^{.ep." data/if_recover/train_sc.log)
    echo "ep=$EP ts=$T -> $R" >> "$LOG"
    ATT=$(echo "$R" | /data/users/zfh/workspace/projects/tarl_reproduction/py310/bin/python \
        -c "import json,sys; print(json.loads(sys.stdin.read())['att'])" 2>/dev/null)
    if [ -n "$ATT" ] && /data/users/zfh/workspace/projects/tarl_reproduction/py310/bin/python \
        -c "import sys; sys.exit(0 if $ATT < $BEST_ATT else 1)"; then
      BEST_ATT=$ATT
      cp /tmp/ckpt_eval_5816.pt "$BEST_CKPT"
      echo "  ^^ new best att=$ATT saved to $BEST_CKPT" >> "$LOG"
    fi
    LAST=$T
  fi
  sleep 30
done
