#!/usr/bin/env bash
# Resubmit every CoLight cell that feeds the journal tables after the
# world-order GNN remap (common/colight_graph.py).
#
# Default `all` is 20 jobs (native L0/L1/L2 + L2 single-axis + rich L1/L2):
#   L0   colight                     prefix m0_homo          (2 nets)
#   L1   colight_odh_l1[_1x21]       prefix l1_odh           (2 nets)
#   L2   colight_odh_l2[_1x21]       prefix l2_odh           (2 nets)
#   axes colight_odh_l2_<axis>[_1x21] prefix l2_axis_<axis>  (5×2)
#   rich colight_rich_odh_l{1,2}     prefix l1_odh / l2_odh  (4)
#
# Prefixes match the extractors. Native and rich share a prefix but write
# different output dirs (sumo_colight_* vs sumo_colight_rich_*).
#
# Confirm the remap is live before sbatch:
#   git pull --no-rebase
#   grep -n remap_graph_edges_to_world common/colight_graph.py agent/colight.py
# After a job starts, logs must contain:
#   [CoLight] remapped N/M graph edges into world intersection order
#
# graph-fix: 18 jobs = all CoLight table cells except L2-axis hetero
# (that pair was already resubmitted after the hub-OD fleet rewrite).
# Compound L2 CoLight is included here (native L2 never reran on the remap).
#
# Old July M0-demand axes (tab:axes4x4, prefixes like hetero_4x4) are a
# different protocol and are not in `all`. Pass `m0-axes` only if you
# intentionally want those.
#
# Slurm: tiny MIG slice + CPU LibSignal. sbatch --mcs-label=...  (NO --export)
#
# Usage (gpujobs):
#   export MCS_LABEL=15288
#   ./extras/submit_colight_table_rerun.sh list
#   ./extras/submit_colight_table_rerun.sh dry-run all
#   ./extras/submit_colight_table_rerun.sh graph-fix
#   ./extras/submit_colight_table_rerun.sh all
#   ./extras/submit_colight_table_rerun.sh l0 l1
#   ./extras/submit_colight_table_rerun.sh axes 4x4
#   ./extras/submit_colight_table_rerun.sh rich
#   ./extras/submit_colight_table_rerun.sh hetero

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"
mkdir -p logs extras/_slurm_generated/colight_fix

SEED="${SEED:-42}"
TIME_RL="${TIME_RL:-48:00:00}"
TIME_SMOKE="${TIME_SMOKE:-02:00:00}"
CPUS_RL="${CPUS_RL:-2}"
MEM_RL="${MEM_RL:-8G}"
GRES="${GRES:-gpu:nvidia_h200_1g.18gb:1}"
PARTITION="${PARTITION:-gpu2}"
AXES=(hetero slow_start crossing_proxy obs noise)

usage() {
  cat <<'EOF'
Usage: ./extras/submit_colight_table_rerun.sh [filters...]

Filters (AND-combined; at least one required):
  list | dry-run | smoke
  all                 20 journal CoLight jobs (L0+L1+L2+axes+rich)
  graph-fix           18 jobs: all except L2-axis hetero (already rerun)
  l0 | l1 | l2 | axes | rich
  4x4 | 1x21
  hetero | slow_start | crossing_proxy | obs | noise
  m0-axes             old July movie-demand axis CoLight (not in `all`)

Examples:
  export MCS_LABEL=15288
  ./extras/submit_colight_table_rerun.sh list
  ./extras/submit_colight_table_rerun.sh graph-fix
  ./extras/submit_colight_table_rerun.sh all
  ./extras/submit_colight_table_rerun.sh axes hetero
  ./extras/submit_colight_table_rerun.sh rich 4x4
EOF
}

# group|short|agent|network|prefix
JOBS=()
add_job() {
  JOBS+=("$1|$2|$3|$4|$5")
}

add_job l0 l0_4x4 colight sumo4x4 m0_homo
add_job l0 l0_i21 colight sumo1x21 m0_homo

add_job l1 l1_4x4 colight_odh_l1 sumo4x4 l1_odh
add_job l1 l1_i21 colight_odh_l1_1x21 sumo1x21 l1_odh

add_job l2 l2_4x4 colight_odh_l2 sumo4x4 l2_odh
add_job l2 l2_i21 colight_odh_l2_1x21 sumo1x21 l2_odh

for axis in "${AXES[@]}"; do
  add_job axes "ax_${axis}_4x4" "colight_odh_l2_${axis}" sumo4x4 "l2_axis_${axis}"
  add_job axes "ax_${axis}_i21" "colight_odh_l2_${axis}_1x21" sumo1x21 "l2_axis_${axis}"
done

add_job rich l1r_4x4 colight_rich_odh_l1 sumo4x4 l1_odh
add_job rich l1r_i21 colight_rich_odh_l1_1x21 sumo1x21 l1_odh
add_job rich l2r_4x4 colight_rich_odh_l2 sumo4x4 l2_odh
add_job rich l2r_i21 colight_rich_odh_l2_1x21 sumo1x21 l2_odh

# Old July M0-demand CoLight (extract_axis_att.py). Not part of `all`.
add_job m0-axes m0ax_het_4x4 colight_hetero sumo4x4 hetero_4x4
add_job m0-axes m0ax_ss_4x4 colight_slow_start sumo4x4 slow_start_4x4
add_job m0-axes m0ax_xp_4x4 colight_crossing_proxy sumo4x4 crossing_proxy_4x4
add_job m0-axes m0ax_full_4x4 colight_realism_full sumo4x4 realism_full_4x4

write_job_script() {
  local short="$1"
  local agent="$2"
  local network="$3"
  local prefix="$4"
  local wall="$5"
  local safe
  safe="$(echo "${agent}_${network}_${prefix}" | tr '/' '_')"
  local out="extras/_slurm_generated/colight_fix/clf_${safe}.sh"

  cat >"${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=clf_${short}
#SBATCH --output=logs/clf_${safe}_%j.out
#SBATCH --error=logs/clf_${safe}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --time=${wall}
#SBATCH --cpus-per-task=${CPUS_RL}
#SBATCH --mem=${MEM_RL}
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
echo "Git:       \$(git rev-parse --short HEAD) \$(git log -1 --format='%s')"
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

"\${PYTHON}" -c "import torch_scatter" 2>/dev/null || {
  TV="\$("\${PYTHON}" -c 'import torch; print(torch.__version__.split("+")[0])')"
  echo "Installing torch_scatter for torch \${TV}..."
  "\${PYTHON}" -m pip install torch_scatter -f "https://data.pyg.org/whl/torch-\${TV}.html"
}

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
echo "Confirm remap: grep '[CoLight] remapped' logs/clf_${safe}_\${SLURM_JOB_ID}.out"
EOF

  chmod +x "${out}"
  echo "${out}"
}

LIST_ONLY=0
DRY_RUN=0
WANT_SMOKE=0
WANT_ALL=0
WANT_GRAPH_FIX=0
SKIP_HETERO=0
SEL_GROUPS=()
SEL_NETWORKS=()
SEL_AXES=()

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
    graph-fix|graphfix) WANT_GRAPH_FIX=1 ;;
    l0|l1|l2|axes|rich|m0-axes) SEL_GROUPS+=("${arg}") ;;
    4x4) SEL_NETWORKS+=(sumo4x4) ;;
    1x21|ingolstadt) SEL_NETWORKS+=(sumo1x21) ;;
    hetero|slow_start|crossing_proxy|obs|noise) SEL_AXES+=("${arg}") ;;
    *)
      echo "Unknown filter: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${WANT_SMOKE} -eq 1 ]]; then
  SEL_GROUPS=(l0)
  SEL_NETWORKS=(sumo4x4)
  TIME_RL="${TIME_SMOKE}"
fi

if [[ ${WANT_ALL} -eq 1 ]]; then
  SEL_GROUPS=(l0 l1 l2 axes rich)
fi
if [[ ${WANT_GRAPH_FIX} -eq 1 ]]; then
  SEL_GROUPS=(l0 l1 l2 axes rich)
  SKIP_HETERO=1
fi

if [[ ${#SEL_GROUPS[@]} -eq 0 ]]; then
  if [[ ${#SEL_AXES[@]} -gt 0 ]]; then
    SEL_GROUPS=(axes)
  elif [[ ${LIST_ONLY} -eq 1 ]]; then
    # `list` with no group shows the journal set (same as `list all`).
    SEL_GROUPS=(l0 l1 l2 axes rich)
  else
    echo "Pick a group (l0|l1|l2|axes|rich|all) or an axis name." >&2
    usage
    exit 1
  fi
fi

match_group() {
  local group="$1"
  local g
  for g in "${SEL_GROUPS[@]}"; do
    [[ "${g}" == "${group}" ]] && return 0
  done
  return 1
}

match_network() {
  local network="$1"
  [[ ${#SEL_NETWORKS[@]} -eq 0 ]] && return 0
  local n
  for n in "${SEL_NETWORKS[@]}"; do
    [[ "${n}" == "${network}" ]] && return 0
  done
  return 1
}

match_axis_job() {
  local group="$1"
  local prefix="$2"
  if [[ ${SKIP_HETERO} -eq 1 && "${prefix}" == "l2_axis_hetero" ]]; then
    return 1
  fi
  [[ "${group}" != "axes" ]] && return 0
  [[ ${#SEL_AXES[@]} -eq 0 ]] && return 0
  local axis
  for axis in "${SEL_AXES[@]}"; do
    [[ "${prefix}" == "l2_axis_${axis}" ]] && return 0
  done
  return 1
}

SELECTED=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r group short agent network prefix <<<"${spec}"
  match_group "${group}" || continue
  match_network "${network}" || continue
  match_axis_job "${group}" "${prefix}" || continue
  SELECTED+=("${spec}")
done

n_jobs=${#SELECTED[@]}
if [[ ${n_jobs} -eq 0 ]]; then
  echo "No jobs matched the filters." >&2
  exit 1
fi

if [[ ${WANT_ALL} -eq 0 && ${WANT_GRAPH_FIX} -eq 0 && ${WANT_SMOKE} -eq 0 && ${n_jobs} -ge 20 && ${LIST_ONLY} -eq 0 && ${DRY_RUN} -eq 0 ]]; then
  echo "Refusing to submit ${n_jobs} jobs without 'all'." >&2
  echo "Narrow the filters, or pass: $0 all" >&2
  exit 1
fi

if [[ ${LIST_ONLY} -eq 0 && ${DRY_RUN} -eq 0 && -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=15288" >&2
  exit 1
fi

need_axis_configs=0
for spec in "${SELECTED[@]}"; do
  IFS='|' read -r group _ _ _ _ <<<"${spec}"
  if [[ "${group}" == "axes" ]]; then
    need_axis_configs=1
    break
  fi
done
if [[ ${need_axis_configs} -eq 1 ]]; then
  python extras/gen_l2_axis_configs.py >/dev/null
fi

echo "CoLight table rerun: ${n_jobs} jobs  seed=${SEED}  gres=${GRES}  HEAD=$(git rev-parse --short HEAD)"
echo "Look for: [CoLight] remapped N/M graph edges into world intersection order"
echo ""
printf "  %-8s %-36s %-10s %s\n" "GROUP" "AGENT" "NETWORK" "PREFIX"
for spec in "${SELECTED[@]}"; do
  IFS='|' read -r group short agent network prefix <<<"${spec}"
  printf "  %-8s %-36s %-10s %s\n" "${group}" "${agent}" "${network}" "${prefix}"
done
echo ""

if [[ ${LIST_ONLY} -eq 1 ]]; then
  echo "Listed ${n_jobs} jobs. Nothing submitted."
  echo "After git pull on gpujobs: export MCS_LABEL=15288 && $0 all"
  exit 0
fi

submitted=0
for spec in "${SELECTED[@]}"; do
  IFS='|' read -r group short agent network prefix <<<"${spec}"
  yml="configs/tsc/${agent}.yml"
  if [[ ! -f "${yml}" ]]; then
    echo "ERROR: missing ${yml}" >&2
    exit 1
  fi
  echo "  ${agent}  ${network}  prefix=${prefix}  wall=${TIME_RL}"
  script="$(write_job_script "${short}" "${agent}" "${network}" "${prefix}" "${TIME_RL}")"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "    wrote ${script}"
  else
    echo "    sbatch ${script}  (mcs=${MCS_LABEL})"
    sbatch --mcs-label="${MCS_LABEL}" "${script}"
  fi
  submitted=$((submitted + 1))
done

echo ""
if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry-run: wrote ${submitted} scripts under extras/_slurm_generated/colight_fix/"
else
  echo "Submitted ${submitted} CoLight table-rerun jobs. gres=${GRES}"
  echo "Monitor: squeue -u \$USER -o '%.8i %.40j %.8T %.10M %.6D %R'"
  echo "Logs: logs/clf_*_<jobid>.out"
  echo "When finished, re-extract tables from data/output_data/tsc/sumo_colight*/"
fi
