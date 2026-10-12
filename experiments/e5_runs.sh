#!/bin/bash
PY=/data/users/zfh/miniconda3/envs/tsc_miss/bin/python
cd /data/users/zfh/workspace/projects/auto_research/experiments
CK=e2_recon
S=9,10,11,12,13
# blackout: whole-approach outage for entire run
for RATE in 0.2 0.4 0.6; do
  for F in last mean zero; do
    $PY eval_closed.py --fill $F --kind blackout --rate $RATE --seeds $S
  done
  $PY eval_closed.py --ckpt $CK --fill model:pred_mean --kind blackout --rate $RATE --seeds $S
  $PY eval_closed.py --ckpt $CK --fill model:staleness:lam=1:tau=2 --kind blackout --rate $RATE --seeds $S
  $PY eval_closed.py --ckpt $CK --fill model:pred_mean --risk lcb --kappa 1.0 --kind blackout --rate $RATE --seeds $S
done
# sustained: k lanes missing for dur decisions — duration sweep
for DUR in 10 50 150; do
  for F in last mean zero; do
    $PY eval_closed.py --fill $F --kind sustained --rate 0.3 --dur $DUR --seeds $S
  done
  $PY eval_closed.py --ckpt $CK --fill model:pred_mean --kind sustained --rate 0.3 --dur $DUR --seeds $S
done
echo DONE
