#!/bin/bash
PY=/data/users/zfh/miniconda3/envs/tsc_miss/bin/python
cd /data/users/zfh/workspace/projects/auto_research/experiments
EP=20
# E2: same arch, different objectives
$PY train_recovery.py --scenario resco4x4 --train-seeds 1,2,3,4,5,6 --val-seeds 7 --epochs $EP --recon nll --decision none --beta 0 --tag e2_recon 2>&1 | tail -30
for D in press rank spo; do
  $PY train_recovery.py --scenario resco4x4 --train-seeds 1,2,3,4,5,6 --val-seeds 7 --epochs $EP --recon nll --decision $D --beta 0.5 --tag e2_$D 2>&1 | tail -30
done
# E1 ablations: temporal-only / spatial-only / full
$PY train_recovery.py --scenario resco4x4 --train-seeds 1,2,3,4,5,6 --val-seeds 7 --epochs $EP --window 1 --recon nll --decision none --beta 0 --tag e1_gnn_only 2>&1 | tail -30
$PY train_recovery.py --scenario resco4x4 --train-seeds 1,2,3,4,5,6 --val-seeds 7 --epochs $EP --layers 0 --recon nll --decision none --beta 0 --tag e1_gru_only 2>&1 | tail -30
$PY train_recovery.py --scenario resco4x4 --train-seeds 1,2,3,4,5,6 --val-seeds 7 --epochs $EP --recon l1 --decision none --beta 0 --tag e1_l1 2>&1 | tail -30
echo DONE
