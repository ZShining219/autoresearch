#!/usr/bin/env bash
# Submit Traffic-R1 L2 single-axis jobs on sumo4x4 only (5 jobs).
#
# Hub OD + one realism_full axis. Inference-only (episodes=1), vLLM, GPU.
# Not the CPU LibSignal path used for DQN/CoLight.
#
# Usage (gpujobs):
#   export MCS_LABEL=15288
#   ./extras/submit_l2_axes_traffic_r1.sh list
#   ./extras/submit_l2_axes_traffic_r1.sh smoke      # noise only
#   ./extras/submit_l2_axes_traffic_r1.sh all        # 5 axes
#   ./extras/submit_l2_axes_traffic_r1.sh noise
#
# Slow-start needs commit 23a8a40 on the server (strip inline pkw).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"
mkdir -p logs extras/_slurm_generated/l2_axes_tr1

SEED="${SEED:-42}"
NETWORK="sumo4x4"
TIME="${TIME:-12:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
GRES="${GRES:-gpu:nvidia_h200_2g.35gb:1}"
PARTITION="${PARTITION:-gpu2}"
# Bake a real env path into the job. Do not call `conda` on the compute node
# (that is why 11385–11389 died in 0s). Prefer an env that already has vllm.
CONDA_PREFIX="${CONDA_PREFIX:-}"
AXES=(hetero slow_start crossing_proxy obs noise)

env_has_vllm() {
  local p="$1"
  [[ -n "${p}" && -x "${p}/bin/python" ]] || return 1
  "${p}/bin/python" -c "import vllm" >/dev/null 2>&1
}

pick_conda_prefix() {
  local p
  if [[ -n "${CONDA_PREFIX:-}" ]] && env_has_vllm "${CONDA_PREFIX}"; then
    echo "${CONDA_PREFIX}"
    return 0
  fi
  for p in \
    /data1/mmirzata/.conda/envs/* \
    "${HOME}/.conda/envs/"* \
    /data1/mmirzata/.conda/envs/libsignal
  do
    [[ "${p}" == /opt/* ]] && continue
    if env_has_vllm "${p}"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage: ./extras/submit_l2_axes_traffic_r1.sh [filters...]

  list | dry-run | smoke | all
  hetero | slow_start | crossing_proxy | obs | noise

Examples:
  export MCS_LABEL=15288
  ./extras/submit_l2_axes_traffic_r1.sh list
  ./extras/submit_l2_axes_traffic_r1.sh all
EOF
}

write_job_script() {
  local axis="$1"
  local cfg="traffic_r1_odh_l2_${axis}"
  local prefix="l2_axis_${axis}"
  local conda_prefix="$2"
  local out="extras/_slurm_generated/l2_axes_tr1/l2ax_tr1_${axis}_sumo4x4.sh"

  cat >"${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=l2a_tr1_${axis}_4x4
#SBATCH --output=logs/l2ax_tr1_${axis}_4x4_%j.out
#SBATCH --error=logs/l2ax_tr1_${axis}_4x4_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --time=${TIME}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd "\${HOME}/LibSignalFork"

CONDA_PREFIX="${conda_prefix}"
for p in /data1/mmirzata/.conda/envs/traffic /data1/mmirzata/.conda/envs/libsignal "\${CONDA_PREFIX}"; do
  [[ -z "\${p}" || ! -x "\${p}/bin/python" ]] && continue
  if "\${p}/bin/python" -c "import vllm" >/dev/null 2>&1; then
    CONDA_PREFIX="\${p}"
    break
  fi
done
export SUMO_HOME="\${SUMO_HOME:-\${CONDA_PREFIX}/share/sumo}"
if [[ ! -d "\${SUMO_HOME}" ]]; then
  SUMO_HOME="\$("\${CONDA_PREFIX}/bin/python" -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')"
  export SUMO_HOME
fi
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"
PYTHON="\${CONDA_PREFIX}/bin/python"

echo "Host:      \$(hostname)"
echo "Job:       \${SLURM_JOB_ID:-local}"
echo "Python:    \${PYTHON} (\$("\${PYTHON}" --version 2>&1))"
echo "CONDA:     \${CONDA_PREFIX}"
echo "SUMO_HOME: \${SUMO_HOME}"
echo "Agent:     ${cfg}"
echo "Axis:      ${axis}"
echo "Network:   ${NETWORK}"
echo "Prefix:    ${prefix}"
echo "Seed:      ${SEED}"
echo "CUDA:      \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start:     \$(date -Is)"

if [[ ! -x "\${PYTHON}" ]]; then
  echo "ERROR: missing \${PYTHON}" >&2
  exit 127
fi

"\${PYTHON}" -c "import vllm, torch, libsumo; print('vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"

"\${PYTHON}" run.py \\
  --agent ${cfg} \\
  --world sumo \\
  --network ${NETWORK} \\
  --seed ${SEED} \\
  --ngpu 0 \\
  --interface libsumo \\
  --prefix ${prefix}

echo "Done: \$(date -Is)"
echo "Metrics: data/output_data/tsc/sumo_${cfg}/${NETWORK}/${prefix}/logger/new_metrics*.csv"
EOF

  chmod +x "${out}"
  echo "${out}"
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

LIST_ONLY=0
DRY_RUN=0
SEL_AXES=()
WANT_ALL=0

for arg in "$@"; do
  case "${arg}" in
    -h|--help|help) usage; exit 0 ;;
    list) LIST_ONLY=1 ;;
    dry-run|dryrun) DRY_RUN=1 ;;
    smoke) SEL_AXES=(noise) ;;
    all) WANT_ALL=1 ;;
    hetero|slow_start|crossing_proxy|obs|noise) SEL_AXES+=("${arg}") ;;
    *)
      echo "Unknown filter: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#SEL_AXES[@]} -eq 0 ]]; then
  SEL_AXES=("${AXES[@]}")
fi

if [[ ${LIST_ONLY} -eq 0 && ${DRY_RUN} -eq 0 && -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288" >&2
  exit 1
fi

if ! CONDA_PREFIX="$(pick_conda_prefix)"; then
  echo "No conda env with vllm found." >&2
  echo "On the login node, find one:" >&2
  echo "  conda env list" >&2
  echo "  ls /data1/mmirzata/.conda/envs" >&2
  echo "  for d in /data1/mmirzata/.conda/envs/*/bin/python; do" >&2
  echo "    echo \"== \$d\"; \"\$d\" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo no vllm" >&2
  echo "  done" >&2
  echo "Then: export CONDA_PREFIX=/path/to/that/env" >&2
  if [[ ${LIST_ONLY} -eq 0 ]]; then
    exit 1
  fi
  CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
fi
echo "Traffic-R1 L2 single-axis 4x4: axes=${SEL_AXES[*]}  gres=${GRES}  conda=${CONDA_PREFIX}"
echo ""

submitted=0
for axis in "${SEL_AXES[@]}"; do
  cfg="traffic_r1_odh_l2_${axis}"
  yml="configs/tsc/${cfg}.yml"
  if [[ ! -f "${yml}" ]]; then
    echo "ERROR: missing ${yml}" >&2
    exit 1
  fi
  echo "  ${cfg}  ${NETWORK}  prefix=l2_axis_${axis}  wall=${TIME}"
  if [[ ${LIST_ONLY} -eq 1 ]]; then
    continue
  fi
  script="$(write_job_script "${axis}" "${CONDA_PREFIX}")"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "    wrote ${script}"
  else
    echo "    sbatch ${script}  (mcs=${MCS_LABEL})"
    sbatch --mcs-label="${MCS_LABEL}" "${script}"
  fi
  submitted=$((submitted + 1))
done

echo ""
if [[ ${LIST_ONLY} -eq 1 ]]; then
  echo "Listed ${#SEL_AXES[@]} jobs. Nothing submitted."
elif [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry-run: wrote ${submitted} scripts under extras/_slurm_generated/l2_axes_tr1/"
else
  echo "Submitted ${submitted} Traffic-R1 L2 axis jobs."
  echo "Monitor: squeue -u \$USER"
fi
