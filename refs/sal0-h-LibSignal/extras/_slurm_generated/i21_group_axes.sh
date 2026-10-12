#!/usr/bin/env bash
#SBATCH --job-name=i21_axes
#SBATCH --output=logs/i21_group_axes_%j.out
#SBATCH --error=logs/i21_group_axes_%j.err
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
echo "Group: axes  Network: ${NETWORK}  Seed: ${SEED}"
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

run_one fixedtime_hetero axis_hetero_1x21_e200
run_one maxpressure_hetero axis_hetero_1x21_e200
run_one dqn_hetero_e200 axis_hetero_1x21_e200
run_one presslight_hetero_e200 axis_hetero_1x21_e200
run_one colight_hetero_e200 axis_hetero_1x21_e200
run_one fixedtime_slow_start axis_slow_start_1x21_e200
run_one maxpressure_slow_start axis_slow_start_1x21_e200
run_one dqn_slow_start_e200 axis_slow_start_1x21_e200
run_one presslight_slow_start_e200 axis_slow_start_1x21_e200
run_one colight_slow_start_e200 axis_slow_start_1x21_e200
run_one fixedtime_crossing_proxy axis_crossing_proxy_1x21_e200
run_one maxpressure_crossing_proxy axis_crossing_proxy_1x21_e200
run_one dqn_crossing_proxy_e200 axis_crossing_proxy_1x21_e200
run_one presslight_crossing_proxy_e200 axis_crossing_proxy_1x21_e200
run_one colight_crossing_proxy_e200 axis_crossing_proxy_1x21_e200
run_one fixedtime_obs axis_obs_1x21_e200
run_one maxpressure_obs axis_obs_1x21_e200
run_one dqn_obs_e200 axis_obs_1x21_e200
run_one presslight_obs_e200 axis_obs_1x21_e200
run_one colight_obs_e200 axis_obs_1x21_e200
run_one fixedtime_noise axis_noise_1x21_e200
run_one maxpressure_noise axis_noise_1x21_e200
run_one dqn_noise_e200 axis_noise_1x21_e200
run_one presslight_noise_e200 axis_noise_1x21_e200
run_one colight_noise_e200 axis_noise_1x21_e200

echo "Group axes finished: $(date -Is)"
