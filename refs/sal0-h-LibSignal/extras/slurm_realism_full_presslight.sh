#!/usr/bin/env bash
#SBATCH --job-name=pl_realism_full_4x4
#SBATCH --output=logs/presslight_realism_full_4x4_%j.out
#SBATCH --error=logs/presslight_realism_full_4x4_%j.err
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
echo "Agent: presslight_realism_full  Network: sumo4x4  Seed: 42"

python run.py -a presslight_realism_full -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix realism_full_4x4
