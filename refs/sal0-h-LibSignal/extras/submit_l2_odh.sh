#!/usr/bin/env bash
# Submit journal L2 cell: OD-hub · axes ON (realism_full) · adaptive early-stop (median wait).
#
# Agents: FixedTime, MaxPressure, DQN, PressLight, CoLight
# Networks: sumo4x4 + sumo1x21
#
# Early-stop (from od_hub_*_1800_base.yml, commit 4fb4d75):
#   cycle_mean of TRAIN median wait over 10 demand files;
#   flat if <5% OR <1s change; stop after 3 flat rotations; cap 500.
#
# Slurm: tiny MIG slice + CPU LibSignal (same lesson as M0).
#   sbatch --mcs-label=... <script>   # NO --export
#
# Usage (gpujobs):
#   export MCS_LABEL=15288
#   ./extras/submit_l2_odh.sh smoke
#   ./extras/submit_l2_odh.sh baselines
#   ./extras/submit_l2_odh.sh rl
#   ./extras/submit_l2_odh.sh all

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/LibSignalFork}"
cd "${REPO_DIR}"
mkdir -p logs extras/_slurm_generated/l2

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288"
  exit 1
fi

MODE="${1:-all}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-l2_odh}"
TIME_BASELINE="${TIME_BASELINE:-02:00:00}"
TIME_RL="${TIME_RL:-48:00:00}"
TIME_SMOKE="${TIME_SMOKE:-02:00:00}"
CPUS="${CPUS:-1}"
MEM="${MEM:-4G}"
CPUS_RL="${CPUS_RL:-2}"
MEM_RL="${MEM_RL:-8G}"
GRES="${GRES:-gpu:nvidia_h200_1g.18gb:1}"
PARTITION="${PARTITION:-gpu2}"

# agent config name, network
# 4x4 uses *_odh_l2 ; 1x21 uses *_odh_l2_1x21
BASELINES_4X4=(fixedtime_odh_l2 maxpressure_odh_l2)
RL_4X4=(dqn_odh_l2 presslight_odh_l2 colight_odh_l2)
BASELINES_1X21=(fixedtime_odh_l2_1x21 maxpressure_odh_l2_1x21)
RL_1X21=(dqn_odh_l2_1x21 presslight_odh_l2_1x21 colight_odh_l2_1x21)

is_rl() {
  case "$1" in
    dqn*|presslight*|colight*) return 0 ;;
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
  local safe
  safe="$(echo "${agent}_${network}" | tr '/' '_')"
  local out="extras/_slurm_generated/l2/l2_${safe}.sh"

  cat >"${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=l2_${agent}
#SBATCH --output=logs/l2_${safe}_%j.out
#SBATCH --error=logs/l2_${safe}_%j.err
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

# MIG for scheduling only; LibSignal on CPU.
export CUDA_VISIBLE_DEVICES=""

"\${PYTHON}" -c "import libsumo; print('libsumo: OK')" 2>/dev/null || {
  echo "WARNING: libsumo missing — may fall back to traci"
}

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

submit_list() {
  local network="$1"
  shift
  local agents=("$@")
  local agent wall cpus mem
  for agent in "${agents[@]}"; do
    if is_rl "${agent}"; then
      wall="${TIME_RL}"
      cpus="${CPUS_RL}"
      mem="${MEM_RL}"
    else
      wall="${TIME_BASELINE}"
      cpus="${CPUS}"
      mem="${MEM}"
    fi
    submit_one "${agent}" "${network}" "${wall}" "${PREFIX}" "${cpus}" "${mem}"
  done
}

case "${MODE}" in
  smoke)
    submit_one maxpressure_odh_l2 sumo4x4 "${TIME_SMOKE}" l2_smoke "${CPUS}" "${MEM}"
    ;;
  baselines)
    submit_list sumo4x4 "${BASELINES_4X4[@]}"
    submit_list sumo1x21 "${BASELINES_1X21[@]}"
    ;;
  rl)
    submit_list sumo4x4 "${RL_4X4[@]}"
    submit_list sumo1x21 "${RL_1X21[@]}"
    ;;
  all)
    submit_list sumo4x4 "${BASELINES_4X4[@]}" "${RL_4X4[@]}"
    submit_list sumo1x21 "${BASELINES_1X21[@]}" "${RL_1X21[@]}"
    ;;
  4x4)
    submit_list sumo4x4 "${BASELINES_4X4[@]}" "${RL_4X4[@]}"
    ;;
  1x21|ingolstadt)
    submit_list sumo1x21 "${BASELINES_1X21[@]}" "${RL_1X21[@]}"
    ;;
  *)
    echo "Usage: $0 {smoke|baselines|rl|all|4x4|1x21}"
    exit 1
    ;;
esac

echo ""
echo "Submitted L2 OD-hub (axes ON (realism_full), median-wait early-stop)."
echo "Prefix=${PREFIX}  gres=${GRES}"
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/l2_*_<jobid>.out"
