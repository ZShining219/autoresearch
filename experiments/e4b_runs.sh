#!/bin/bash
PY=/data/users/zfh/miniconda3/envs/tsc_miss/bin/python
cd /data/users/zfh/workspace/projects/auto_research/experiments
for KIND in correlated random; do
 for RATE in 0.3 0.5; do
  $PY eval_closed.py --ckpt e2_spo --fill model:staleness:lam=1:tau=2 --kind $KIND --rate $RATE --seeds 9,10,11,12,13
  $PY eval_closed.py --ckpt e2_spo --fill model:pred_mean --risk lcb --kappa 1.0 --kind $KIND --rate $RATE --seeds 9,10,11,12,13
 done
done
# kappa sweep on corr0.5 (strongest missingness)
for K in 0.5 2.0 4.0; do
  $PY eval_closed.py --ckpt e2_recon --fill model:pred_mean --risk lcb --kappa $K --kind correlated --rate 0.5 --seeds 9,10,11,12,13
done
echo DONE
