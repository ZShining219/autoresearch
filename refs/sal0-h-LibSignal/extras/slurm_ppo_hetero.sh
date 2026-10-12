#!/usr/bin/env bash
#SBATCH --job-name=ippo_hetero_4x4
#SBATCH --output=logs/ippo_hetero_4x4_%j.out
#SBATCH --error=logs/ippo_hetero_4x4_%j.err
#SBATCH --time=24:00:00
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
echo "Agent: ppo_pfrl_hetero  Network: sumo4x4  Seed: 42"

python run.py -a ppo_pfrl_hetero -w sumo -n sumo4x4 --seed 42 --interface libsumo --prefix hetero_test
