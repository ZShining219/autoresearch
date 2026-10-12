#!/bin/bash
PY=/data/users/zfh/miniconda3/envs/tsc_miss/bin/python
cd /data/users/zfh/workspace/projects/auto_research/experiments
CK=e2_recon
# E3: fixed model params, vary decision rule — correlated@0.5 + random@0.5 + correlated@0.3
for KIND in correlated random; do
 for RATE in 0.3 0.5; do
  $PY eval_closed.py --ckpt $CK --fill model:pred_mean --risk none --kind $KIND --rate $RATE --seeds 9,10,11,12,13
  $PY eval_closed.py --ckpt $CK --fill model:fixed_disc:lam=1 --risk none --kind $KIND --rate $RATE --seeds 9,10,11,12,13
  $PY eval_closed.py --ckpt $CK --fill model:staleness:lam=1:tau=2 --risk none --kind $KIND --rate $RATE --seeds 9,10,11,12,13
  $PY eval_closed.py --ckpt $CK --fill model:pred_mean --risk lcb --kappa 1.0 --kind $KIND --rate $RATE --seeds 9,10,11,12,13
 done
done
echo DONE
