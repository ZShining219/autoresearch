#!/usr/bin/env bash
# Submit M0 journal cell: movie · axes OFF · 3600s · 200 eps (default configs).
#
# gpujobs lesson: CPU-only jobs on partition gpu2 sit forever at (Priority) while
# GPU jobs wait. Request a tiny MIG slice so Slurm will schedule; LibSignal still
# runs on CPU (--ngpu -1).
#
#   sbatch --mcs-label=... <script>     # NO --export
#
# Usage (gpujobs login):
#   export MCS_LABEL=15288
#   ./extras/submit_m0_homo.sh smoke      # 1 short MP 4x4 job
#   ./extras/submit_m0_homo.sh baselines
#   ./extras/submit_m0_homo.sh rl
#   ./extras/submit_m0_homo.sh all

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/LibSignalFork}"
cd "${REPO_DIR}"
mkdir -p logs extras/_slurm_generated/m0

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288"
  exit 1
fi

MODE="${1:-all}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-m0_homo}"
TIME_BASELINE="${TIME_BASELINE:-04:00:00}"
TIME_RL="${TIME_RL:-48:00:00}"
TIME_SMOKE="${TIME_SMOKE:-02:00:00}"
# Match the working dry-run that started immediately
CPUS="${CPUS:-1}"
MEM="${MEM:-4G}"
GRES="${GRES:-gpu:nvidia_h200_1g.18gb:1}"
PARTITION="${PARTITION:-gpu2}"

BASELINES=(fixedtime maxpressure)
RL=(dqn presslight colight)

is_rl() {
  case "$1" in
    dqn|presslight|colight) return 0 ;;
    *) return 1 ;;
  esac
}

write_job_script() {
  local agent="$1"
  local network="$2"
  local wall="$3"
  local prefix="$4"
  local cpus="$5"
  local mem="$6"
  local out="extras/_slurm_generated/m0/m0_${agent}_${network}.sh"

  cat >"${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=m0_${agent}_${network}
#SBATCH --output=logs/m0_${agent}_${network}_%j.out
#SBATCH --error=logs/m0_${agent}_${network}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --time=${wall}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd "\${HOME}/LibSignalFork"

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"
PYTHON="\${CONDA_PREFIX}/bin/python"

echo "Host:      \$(hostname)"
echo "Job:       \${SLURM_JOB_ID:-local}"
echo "Python:    \${PYTHON} (\$("\${PYTHON}" --version 2>&1))"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo "Agent:     ${agent}"
echo "Network:   ${network}"
echo "Seed:      ${SEED}"
echo "Prefix:    ${prefix}"
echo "Start:     \$(date -Is)"

if [[ ! -x "\${PYTHON}" ]]; then
  echo "ERROR: missing \${PYTHON}" >&2
  exit 127
fi

if [[ "${agent}" == colight* ]]; then
  "\${PYTHON}" -c "import torch_scatter" 2>/dev/null || {
    TV="\$("\${PYTHON}" -c 'import torch; print(torch.__version__.split("+")[0])')"
    echo "Installing torch_scatter for torch \${TV}..."
    "\${PYTHON}" -m pip install torch_scatter -f "https://data.pyg.org/whl/torch-\${TV}.html"
  }
fi

# GPU MIG slice is for Slurm scheduling only. Force CPU for LibSignal even if
# Slurm exported CUDA_VISIBLE_DEVICES=MIG-... (--ngpu -1 clears it in run.py too).
export CUDA_VISIBLE_DEVICES=""

"\${PYTHON}" -c "import libsumo; print('libsumo: OK')" 2>/dev/null || {
  echo "WARNING: libsumo missing — may fall back to traci"
}

# LibSignal on CPU (--ngpu -1).
"\${PYTHON}" run.py \\
  --agent ${agent} \\
  --world sumo \\
  --network ${network} \\
  --seed ${SEED} \\
  --ngpu -1 \\
  --interface libsumo \\
  --prefix ${prefix}

echo "Done: \$(date -Is)"
echo "Metrics: data/output_data/tsc/sumo_${agent}/${network}/${prefix}/logger/new_metrics*.csv"
EOF

  chmod +x "${out}"
  echo "${out}"
}

submit_one() {
  local agent="$1"
  local network="$2"
  local wall="$3"
  local prefix="${4:-${PREFIX}}"
  local cpus="${5:-${CPUS}}"
  local mem="${6:-${MEM}}"
  local script
  script="$(write_job_script "${agent}" "${network}" "${wall}" "${prefix}" "${cpus}" "${mem}")"
  echo "sbatch ${script}  (mcs=${MCS_LABEL}, gres=${GRES}, cpus=${cpus}, mem=${mem})"
  sbatch --mcs-label="${MCS_LABEL}" "${script}"
}

submit_net() {
  local network="$1"
  shift
  local agents=("$@")
  local agent wall cpus mem
  for agent in "${agents[@]}"; do
    if is_rl "${agent}"; then
      wall="${TIME_RL}"
      cpus="${CPUS_RL:-2}"
      mem="${MEM_RL:-8G}"
    else
      wall="${TIME_BASELINE}"
      cpus="${CPUS}"
      mem="${MEM}"
    fi
    submit_one "${agent}" "${network}" "${wall}" "${PREFIX}" "${cpus}" "${mem}"
  done
}

ALL_AGENTS=("${BASELINES[@]}" "${RL[@]}")

case "${MODE}" in
  smoke)
    submit_one maxpressure sumo4x4 "${TIME_SMOKE}" m0_smoke "${CPUS}" "${MEM}"
    ;;
  baselines)
    submit_net sumo4x4 "${BASELINES[@]}"
    submit_net sumo1x21 "${BASELINES[@]}"
    ;;
  rl)
    submit_net sumo4x4 "${RL[@]}"
    submit_net sumo1x21 "${RL[@]}"
    ;;
  colight)
    submit_one colight sumo4x4 "${TIME_RL}" "${PREFIX}" "${CPUS_RL:-2}" "${MEM_RL:-8G}"
    submit_one colight sumo1x21 "${TIME_RL}" "${PREFIX}" "${CPUS_RL:-2}" "${MEM_RL:-8G}"
    ;;
  all)
    submit_net sumo4x4 "${ALL_AGENTS[@]}"
    submit_net sumo1x21 "${ALL_AGENTS[@]}"
    ;;
  4x4)
    submit_net sumo4x4 "${ALL_AGENTS[@]}"
    ;;
  1x21|ingolstadt)
    submit_net sumo1x21 "${ALL_AGENTS[@]}"
    ;;
  *)
    echo "Usage: $0 {smoke|baselines|rl|colight|all|4x4|1x21}"
    exit 1
    ;;
esac

echo ""
echo "Submitted with gres=${GRES} (CPU compute via --ngpu -1)."
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/m0_<agent>_<network>_<jobid>.out"
