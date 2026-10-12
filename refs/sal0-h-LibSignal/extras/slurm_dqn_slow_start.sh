#!/usr/bin/env bash
#SBATCH --job-name=dqn_slow_start_4x4
#SBATCH --output=logs/dqn_slow_start_4x4_%j.out
#SBATCH --error=logs/dqn_slow_start_4x4_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd ~/LibSignalFork

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "Host: $(hostname)"
echo "Agent: dqn_slow_start  Network: sumo4x4  Seed: 42"

python run.py -a dqn_slow_start -w sumo -n sumo4x4 --seed 42 --interface libsumo --prefix slow_start_4x4
