#!/usr/bin/env bash
#SBATCH --job-name=colight_hetero_4x4
#SBATCH --output=logs/colight_hetero_4x4_%j.out
#SBATCH --error=logs/colight_hetero_4x4_%j.err
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
echo "Agent: colight_hetero  Network: sumo4x4  Seed: 42"

# Attempt to install torch_scatter if missing
python -c "import torch_scatter" 2>/dev/null || {
  echo "torch_scatter not found, attempting install..."
  pip install --no-build-isolation torch_scatter || {
    echo "WARNING: torch_scatter install failed. CoLight will not run."
    exit 1
  }
}

python run.py -a colight_hetero -w sumo -n sumo4x4 --seed 42 --interface libsumo --prefix hetero_test
