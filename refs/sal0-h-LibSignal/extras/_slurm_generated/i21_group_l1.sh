#!/usr/bin/env bash
#SBATCH --job-name=i21_l1
#SBATCH --output=logs/i21_group_l1_%j.out
#SBATCH --error=logs/i21_group_l1_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
cd ~/LibSignalFork

NETWORK="sumo1x21"
SEED="42"

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "Host: $(hostname)"
echo "Group: l1  Network: ${NETWORK}  Seed: ${SEED}"
echo "Start: $(date -Is)"

run_one() {
  local agent="$1"
  local prefix="$2"
  echo "===== START agent=${agent} prefix=${prefix} $(date -Is) ====="
  if [[ "${agent}" == colight* ]]; then
    python -c "import torch_scatter" 2>/dev/null || {
      TV="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
      pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TV}.html"
    }
  fi
  python run.py \
    -a "${agent}" \
    -w sumo \
    -n "${NETWORK}" \
    --seed "${SEED}" \
    --ngpu -1 \
    --interface libsumo \
    --prefix "${prefix}"
  echo "===== DONE  agent=${agent} prefix=${prefix} $(date -Is) ====="
}

run_one fixedtime_odh_l1_1x21 odh_l1_1x21_es
run_one maxpressure_odh_l1_1x21 odh_l1_1x21_es
run_one dqn_odh_l1_1x21 odh_l1_1x21_es
run_one presslight_odh_l1_1x21 odh_l1_1x21_es
run_one colight_odh_l1_1x21 odh_l1_1x21_es

echo "Group l1 finished: $(date -Is)"
