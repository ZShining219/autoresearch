#!/bin/bash
PY=/data/users/zfh/miniconda3/envs/tsc_miss/bin/python
cd /data/users/zfh/workspace/projects/auto_research/experiments
# E4: same arch different training objectives, closed loop
for CK in e2_press e2_rank e2_spo; do
 for KIND in correlated random; do
  for RATE in 0.3 0.5; do
   $PY eval_closed.py --ckpt $CK --fill model:pred_mean --risk none --kind $KIND --rate $RATE --seeds 9,10,11,12,13
  done
 done
done
echo DONE
