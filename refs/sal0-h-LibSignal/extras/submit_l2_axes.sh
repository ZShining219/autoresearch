#!/usr/bin/env bash
# Submit L2 single-axis ablations: hub OD · ONE realism_full axis · L1/L2 early-stop.
#
# This is not the old M0 single-axis sweep (fixed 200 ep, p=0.1 obs, no hub OD).
# It is L2 with four axes turned off:
#   demand/early-stop  = od_hub_*_1800_base.yml (cycle-mean median wait, cap 500)
#   axis parameters    = realism_full_world.yml (obs p=0.8, noise sigma=2, ...)
#   seed               = 42
#
# Agents: FixedTime, MaxPressure, DQN, PressLight, CoLight
# Networks: sumo4x4 + sumo1x21
# Axes: hetero, slow_start, crossing_proxy, obs, noise
#
# Full matrix is 5 axes × 5 agents × 2 networks = 50 jobs. Do not pass `all`
# unless you mean that. Filters AND together.
#
# Default networks are both sumo4x4 and sumo1x21 (Ingolstadt). Pass 4x4 or
# 1x21 only if you want to restrict.
#
# Usage (gpujobs):
#   export MCS_LABEL=15288
#   ./extras/submit_l2_axes.sh list noise
#   ./extras/submit_l2_axes.sh dry-run noise
#   ./extras/submit_l2_axes.sh smoke
#   ./extras/submit_l2_axes.sh noise            # both nets, all 5 agents
#   ./extras/submit_l2_axes.sh rl noise         # both nets, DQN/PressLight/CoLight
#   ./extras/submit_l2_axes.sh colight noise    # both nets, CoLight only
#   ./extras/submit_l2_axes.sh 4x4              # all axes, grid only
#   ./extras/submit_l2_axes.sh 1x21             # all axes, Ingolstadt only
#   ./extras/submit_l2_axes.sh all              # 50 jobs, both nets, all axes
#
# Slurm: tiny MIG slice + CPU LibSignal. sbatch --mcs-label=...  (NO --export)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"
mkdir -p logs extras/_slurm_generated/l2_axes

SEED="${SEED:-42}"
TIME_BASELINE="${TIME_BASELINE:-02:00:00}"
TIME_RL="${TIME_RL:-48:00:00}"
TIME_SMOKE="${TIME_SMOKE:-02:00:00}"
CPUS="${CPUS:-1}"
MEM="${MEM:-4G}"
CPUS_RL="${CPUS_RL:-2}"
MEM_RL="${MEM_RL:-8G}"
GRES="${GRES:-gpu:nvidia_h200_1g.18gb:1}"
PARTITION="${PARTITION:-gpu2}"

AXES=(hetero slow_start crossing_proxy obs noise)
AGENTS_BASELINE=(fixedtime maxpressure)
AGENTS_RL=(dqn presslight colight)
NETWORKS=(sumo4x4 sumo1x21)

usage() {
  cat <<'EOF'
Usage: ./extras/submit_l2_axes.sh [filters...]

Filters (AND-combined; at least one required):
  list              print the selected jobs and exit
  dry-run           write slurm scripts but do not sbatch
  smoke             MaxPressure + noise on 4x4 only
  all               5 axes × 5 agents × 2 networks
  baselines | rl
  4x4 | 1x21
  hetero | slow_start | crossing_proxy | obs | noise
  fixedtime | maxpressure | dqn | presslight | colight

Examples:
  export MCS_LABEL=15288
  ./extras/submit_l2_axes.sh list noise          # both nets
  ./extras/submit_l2_axes.sh smoke               # 4x4 MaxPressure + noise
  ./extras/submit_l2_axes.sh noise               # both nets, all 5 agents
  ./extras/submit_l2_axes.sh rl noise            # both nets, RL only
  ./extras/submit_l2_axes.sh 1x21 noise          # Ingolstadt only
EOF
}

is_rl() {
  case "$1" in
    dqn*|presslight*|colight*) return 0 ;;
    *) return 1 ;;
  esac
}

agent_config() {
  local agent="$1"
  local axis="$2"
  local network="$3"
  if [[ "${network}" == "sumo1x21" ]]; then
    echo "${agent}_odh_l2_${axis}_1x21"
  else
    echo "${agent}_odh_l2_${axis}"
  fi
}

run_prefix() {
  echo "l2_axis_$1"
}

write_job_script() {
  local cfg="$1"
  local network="$2"
  local wall="$3"
  local prefix="$4"
  local cpus="$5"
  local mem="$6"
  local axis="$7"
  local agent="$8"
  local net_tag
  case "${network}" in
    sumo4x4) net_tag="4x4" ;;
    sumo1x21) net_tag="i21" ;;
    *) net_tag="${network}" ;;
  esac
  local safe
  safe="$(echo "${cfg}_${network}" | tr '/' '_')"
  local out="extras/_slurm_generated/l2_axes/l2ax_${safe}.sh"

  cat >"${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=l2a_${agent}_${axis}_${net_tag}
#SBATCH --output=logs/l2ax_${safe}_%j.out
#SBATCH --error=logs/l2ax_${safe}_%j.err
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
echo "Agent:     ${cfg}"
echo "Network:   ${network}"
echo "Axis:      ${axis}"
echo "Seed:      ${SEED}"
echo "Prefix:    ${prefix}"
echo "Start:     \$(date -Is)"

if [[ ! -x "\${PYTHON}" ]]; then
  echo "ERROR: missing \${PYTHON}" >&2
  exit 127
fi

if [[ "${cfg}" == colight* ]]; then
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
  --agent ${cfg} \\
  --world sumo \\
  --network ${network} \\
  --seed ${SEED} \\
  --ngpu -1 \\
  --interface libsumo \\
  --prefix ${prefix}

echo "Done: \$(date -Is)"
echo "Metrics: data/output_data/tsc/sumo_${cfg}/${network}/${prefix}/logger/new_metrics*.csv"
EOF

  chmod +x "${out}"
  echo "${out}"
}

LIST_ONLY=0
DRY_RUN=0
WANT_SMOKE=0
WANT_ALL=0
WANT_BASELINES=0
WANT_RL=0
SEL_AXES=()
SEL_AGENTS=()
SEL_NETWORKS=()

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for arg in "$@"; do
  case "${arg}" in
    -h|--help|help) usage; exit 0 ;;
    list) LIST_ONLY=1 ;;
    dry-run|dryrun) DRY_RUN=1 ;;
    smoke) WANT_SMOKE=1 ;;
    all) WANT_ALL=1 ;;
    baselines) WANT_BASELINES=1 ;;
    rl) WANT_RL=1 ;;
    4x4) SEL_NETWORKS+=(sumo4x4) ;;
    1x21|ingolstadt) SEL_NETWORKS+=(sumo1x21) ;;
    hetero|slow_start|crossing_proxy|obs|noise) SEL_AXES+=("${arg}") ;;
    fixedtime|maxpressure|dqn|presslight|colight) SEL_AGENTS+=("${arg}") ;;
    *)
      echo "Unknown filter: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${WANT_SMOKE} -eq 1 ]]; then
  SEL_AXES=(noise)
  SEL_AGENTS=(maxpressure)
  SEL_NETWORKS=(sumo4x4)
  TIME_BASELINE="${TIME_SMOKE}"
fi

if [[ ${#SEL_AXES[@]} -eq 0 ]]; then
  SEL_AXES=("${AXES[@]}")
fi
if [[ ${#SEL_NETWORKS[@]} -eq 0 ]]; then
  SEL_NETWORKS=("${NETWORKS[@]}")
fi
if [[ ${#SEL_AGENTS[@]} -eq 0 ]]; then
  if [[ ${WANT_BASELINES} -eq 1 && ${WANT_RL} -eq 0 ]]; then
    SEL_AGENTS=("${AGENTS_BASELINE[@]}")
  elif [[ ${WANT_RL} -eq 1 && ${WANT_BASELINES} -eq 0 ]]; then
    SEL_AGENTS=("${AGENTS_RL[@]}")
  else
    SEL_AGENTS=("${AGENTS_BASELINE[@]}" "${AGENTS_RL[@]}")
  fi
elif [[ ${WANT_BASELINES} -eq 1 || ${WANT_RL} -eq 1 ]]; then
  filtered=()
  for agent in "${SEL_AGENTS[@]}"; do
    if is_rl "${agent}"; then
      [[ ${WANT_RL} -eq 1 || ${WANT_BASELINES} -eq 0 ]] && filtered+=("${agent}")
    else
      [[ ${WANT_BASELINES} -eq 1 || ${WANT_RL} -eq 0 ]] && filtered+=("${agent}")
    fi
  done
  SEL_AGENTS=("${filtered[@]}")
fi

# Refuse the full 50-job matrix unless `all` was explicit (smoke/list/dry-run ok).
n_axes=${#SEL_AXES[@]}
n_agents=${#SEL_AGENTS[@]}
n_nets=${#SEL_NETWORKS[@]}
n_jobs=$((n_axes * n_agents * n_nets))
if [[ ${WANT_ALL} -eq 0 && ${WANT_SMOKE} -eq 0 && ${n_jobs} -ge 50 && ${LIST_ONLY} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
  echo "Refusing to submit the full ${n_jobs}-job matrix without 'all'." >&2
  echo "Narrow with an axis/agent/network, or pass: $0 all" >&2
  exit 1
fi

if [[ ${LIST_ONLY} -eq 0 && ${DRY_RUN} -eq 0 && -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288" >&2
  exit 1
fi

python extras/gen_l2_axis_configs.py >/dev/null

echo "L2 single-axis selection: axes=${SEL_AXES[*]}  agents=${SEL_AGENTS[*]}  nets=${SEL_NETWORKS[*]}  jobs=${n_jobs}"
echo "seed=${SEED}  gres=${GRES}  dry_run=${DRY_RUN}"
echo ""

submitted=0
for axis in "${SEL_AXES[@]}"; do
  prefix="$(run_prefix "${axis}")"
  for network in "${SEL_NETWORKS[@]}"; do
    for agent in "${SEL_AGENTS[@]}"; do
      cfg="$(agent_config "${agent}" "${axis}" "${network}")"
      yml="configs/tsc/${cfg}.yml"
      if [[ ! -f "${yml}" ]]; then
        echo "ERROR: missing ${yml}" >&2
        exit 1
      fi
      if is_rl "${agent}"; then
        wall="${TIME_RL}"
        cpus="${CPUS_RL}"
        mem="${MEM_RL}"
      else
        wall="${TIME_BASELINE}"
        cpus="${CPUS}"
        mem="${MEM}"
      fi
      echo "  ${cfg}  ${network}  prefix=${prefix}  wall=${wall}"
      if [[ ${LIST_ONLY} -eq 1 ]]; then
        continue
      fi
      script="$(write_job_script "${cfg}" "${network}" "${wall}" "${prefix}" "${cpus}" "${mem}" "${axis}" "${agent}")"
      if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "    wrote ${script}"
      else
        echo "    sbatch ${script}  (mcs=${MCS_LABEL})"
        sbatch --mcs-label="${MCS_LABEL}" "${script}"
      fi
      submitted=$((submitted + 1))
    done
  done
done

echo ""
if [[ ${LIST_ONLY} -eq 1 ]]; then
  echo "Listed ${n_jobs} jobs. Nothing submitted."
elif [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry-run: wrote ${submitted} scripts under extras/_slurm_generated/l2_axes/"
else
  echo "Submitted ${submitted} L2 single-axis jobs. Prefix=l2_axis_<axis>  gres=${GRES}"
  echo "Monitor: squeue -u \$USER"
  echo "Logs: logs/l2ax_*_<jobid>.out"
fi
