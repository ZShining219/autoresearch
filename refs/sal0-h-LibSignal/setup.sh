#!/usr/bin/env bash
# ============================================================================
# LibSignal Setup Script
# Run this after cloning the repo on a fresh server:
#   chmod +x setup.sh && ./setup.sh
# ============================================================================
set -euo pipefail

# command‑line options
NO_SUDO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sudo) NO_SUDO=1; shift ;;  # skip any operations requiring sudo
        *) break ;;
    esac
done

CONDA_ENV="traffic"
PYTHON_VERSION="3.10"
SUMO_VERSION="1.26.0"

# Usage notes:
#   run as a non-root user after an administrator has preinstalled system
#   packages; supply --no-sudo to skip any privileged steps.
#
# To prevent unbounded CPU/RAM use on a shared server you can launch the
# training/evaluation commands inside a container or cgroup, or use shell
# ulimit(1) settings.  The script itself does not manage resource caps.

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Install Miniconda if not present ─────────────────────────────────────
install_conda() {
    # Check if conda is on PATH or installed at common locations
    if command -v conda &>/dev/null; then
        info "Conda already installed: $(conda --version)"
        return 0
    fi

    # Check common install locations (conda may exist but not be on PATH yet)
    local conda_paths=("${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" "/opt/conda")
    for cpath in "${conda_paths[@]}"; do
        if [ -f "${cpath}/bin/conda" ]; then
            info "Found existing conda at ${cpath} (not on PATH). Initializing..."
            eval "$("${cpath}/bin/conda" shell.bash hook)"
            conda init bash 2>/dev/null || true
            conda init zsh  2>/dev/null || true
            info "Conda initialized: $(conda --version)"
            return 0
        fi
    done

    info "Installing Miniconda..."
    local os_name arch installer
    os_name="$(uname -s)"
    arch="$(uname -m)"
    case "${os_name}" in
        Linux)  os_name="Linux" ;;
        Darwin) os_name="MacOSX" ;;
        *)      error "Unsupported OS: ${os_name}" ;;
    esac
    # Normalize arch (aarch64 -> arm64 for macOS)
    case "${arch}" in
        aarch64) arch="arm64" ;;
    esac
    installer="Miniconda3-latest-${os_name}-${arch}.sh"
    info "Downloading ${installer}..."
    curl -fsSL "https://repo.anaconda.com/miniconda/${installer}" -o "/tmp/${installer}"
    bash "/tmp/${installer}" -b -p "${HOME}/miniconda3"
    rm -f "/tmp/${installer}"

    # Init conda for current shell
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    conda init bash 2>/dev/null || true
    conda init zsh  2>/dev/null || true
    info "Miniconda installed at ${HOME}/miniconda3"
}

# ── 2. Create conda environment ────────────────────────────────────────────
create_env() {
    # Initialize conda for this shell session (required for non-interactive scripts)
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
    source "${conda_base}/etc/profile.d/conda.sh"

    # Accept Anaconda TOS if required (conda >= 25.x)
    if conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &>/dev/null; then
        true
    fi
    if conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r &>/dev/null; then
        true
    fi

    if conda env list | grep -qw "^${CONDA_ENV} "; then
        warn "Conda env '${CONDA_ENV}' already exists. Skipping creation."
    else
        info "Creating conda env '${CONDA_ENV}' with Python ${PYTHON_VERSION}..."
        conda create -y -n "${CONDA_ENV}" -c conda-forge python="${PYTHON_VERSION}"
    fi

    info "Activating '${CONDA_ENV}'..."
    conda activate "${CONDA_ENV}"
    info "Active Python: $(which python) ($(python --version))"

    # Ensure pip lives inside the conda env (not system/user pip on PATH)
    if ! python -m pip --version &>/dev/null; then
        info "Installing pip into '${CONDA_ENV}'..."
        conda install -y -n "${CONDA_ENV}" pip
    fi
    info "Active pip: $(python -m pip --version)"
}

# ── 3. Install PyTorch (with CUDA if available) ────────────────────────────
install_pytorch() {
    info "Detecting GPU..."
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # macOS: default PyTorch supports MPS (Apple Silicon) automatically
        info "macOS detected. Installing PyTorch (MPS will be used on Apple Silicon)..."
        python -m pip install torch torchvision
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local cuda_version
        cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        info "NVIDIA GPU detected (driver: ${cuda_version})"
        info "Installing PyTorch with CUDA support..."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        warn "No NVIDIA GPU detected. Installing CPU-only PyTorch."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
}

# ── 4. Install torch-geometric + torch-scatter ─────────────────────────────
install_torch_geometric() {
    info "Installing torch-geometric..."
    python -m pip install torch-geometric

    info "Installing torch-scatter (required for CoLight agent)..."
    # Use conda for torch-scatter as pip installation often fails
    conda install torch-scatter=2.1.1 -c conda-forge -y

    # Install pyg-lib for optimized ops (optional, best-effort)
    python -m pip install pyg-lib -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__.split("+")[0])')+$(python -c 'import torch; print("cu121" if torch.cuda.is_available() else "cpu")').html" 2>/dev/null \
        || warn "pyg-lib prebuilt wheel not available — torch-geometric will still work fine"
}

# ── 5. Install SUMO packages (aligned versions) ────────────────────────────
install_sumo() {
    info "Installing SUMO packages (version ${SUMO_VERSION})..."

    if [[ "$(uname -s)" == "Darwin" ]]; then
        warn "macOS detected: libsumo wheels not available for macOS."
        warn "Install SUMO via Homebrew: brew install sumo"
        warn "The framework will use traci (slower but compatible)."
        python -m pip install sumolib=="${SUMO_VERSION}" traci=="${SUMO_VERSION}"
    else
        # Uninstall any existing SUMO packages first to avoid version conflicts
        info "Cleaning existing SUMO packages..."
        python -m pip uninstall -y libsumo sumolib traci eclipse-sumo 2>/dev/null || true

        # Install libsumo FIRST, then align the rest to the same version
        info "Installing libsumo==${SUMO_VERSION}..."
        python -m pip install libsumo=="${SUMO_VERSION}"

        info "Installing sumolib, traci, eclipse-sumo (force-reinstall to align versions)..."
        python -m pip install --force-reinstall sumolib=="${SUMO_VERSION}" traci=="${SUMO_VERSION}" eclipse-sumo=="${SUMO_VERSION}"

        # Verify libsumo actually loads (it needs system libs like libGL on some distros)
        if python -c "import libsumo" 2>/dev/null; then
            info "libsumo verified — fast simulation mode available"
        else
            warn "libsumo installed but won't load. Installing system dependencies..."
            # Common missing libs on Ubuntu/Debian
            if command -v apt-get &>/dev/null; then
                if [ "$NO_SUDO" -eq 1 ]; then
                    warn "would install system packages (libgl, etc.) here but --no-sudo was specified."
                    warn "ask a sysadmin to install them or run this script again with sudo."
                else
                    sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 libglib2.0-0 libxrender1 libfontconfig1 2>/dev/null || true
                fi
            fi
            # Re-check
            if python -c "import libsumo" 2>/dev/null; then
                info "libsumo verified after installing system deps"
            else
                warn "libsumo still won't load. Use --interface traci as fallback."
            fi
        fi
    fi

    # Set SUMO_HOME to the pip-installed eclipse-sumo
    local sumo_home
    sumo_home=$(python -c "import os, sumolib; print(os.path.dirname(os.path.dirname(sumolib.__file__)))")
    info "SUMO_HOME will be set to: ${sumo_home}"

    # Persist SUMO_HOME in conda env activation
    mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
    mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

    cat > "${CONDA_PREFIX}/etc/conda/activate.d/sumo_env.sh" << EOF
export SUMO_HOME="${sumo_home}"
EOF

    cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/sumo_env.sh" << EOF
unset SUMO_HOME
EOF

    # Set it for the current session too
    export SUMO_HOME="${sumo_home}"
}

# ── 6. Install remaining Python dependencies ───────────────────────────────
install_deps() {
    info "Installing Python dependencies..."
    python -m pip install \
        "gym>=0.21.0" \
        "gymnasium>=0.26.0" \
        "numpy<2.0" \
        "lmdb>=1.3.0" \
        "PyYAML>=6.0" \
        "pfrl>=0.3.0" \
        "stable-baselines3>=1.5.0" \
        "matplotlib>=3.5.0" \
        "pandas>=1.3.0" \
        "seaborn>=0.11.0" \
        "tqdm>=4.62.0" \
        "mpmath>=1.2.1" \
        "sympy>=1.10.1"
}

# ── 7. Verify installation ─────────────────────────────────────────────────
verify() {
    info "Verifying installation..."
    local failed=0

    python -c "import torch; print(f'  PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')" || { warn "PyTorch import failed"; failed=1; }
    python -c "import torch_geometric; print(f'  torch-geometric {torch_geometric.__version__}')" || { warn "torch-geometric import failed"; failed=1; }
    python -c "import torch_scatter; print(f'  torch-scatter OK')" || { warn "torch-scatter import failed"; failed=1; }
    python -c "import pfrl; print(f'  pfrl {pfrl.__version__}')" || { warn "pfrl import failed"; failed=1; }
    if [[ "$(uname -s)" == "Darwin" ]]; then
        warn "libsumo: skipped (not available on macOS — using traci instead)"
    else
        python -c "import libsumo; print(f'  libsumo OK')" || { warn "libsumo import failed"; failed=1; }
    fi
    python -c "import sumolib; print(f'  sumolib OK')" || { warn "sumolib import failed"; failed=1; }
    python -c "import traci; print(f'  traci OK')" || { warn "traci import failed"; failed=1; }
    python -c "import gym; print(f'  gym {gym.__version__}')" || { warn "gym import failed"; failed=1; }
    python -c "import numpy; print(f'  numpy {numpy.__version__}')" || { warn "numpy import failed"; failed=1; }

    if [ "${failed}" -eq 0 ]; then
        info "All imports verified successfully!"
    else
        warn "Some imports failed — check warnings above."
    fi
}

# ── 8. Print summary ───────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "============================================="
    echo -e "${GREEN}  LibSignal setup complete!${NC}"
    echo "============================================="
    echo ""
    echo "  Conda env:  ${CONDA_ENV}"
    echo "  Activate:   conda activate ${CONDA_ENV}"
    echo ""
    echo "  Example run:"
    echo "    python run.py --task tsc --agent dqn --world sumo --network sumo1x1 --dataset onfly"
    echo ""
    echo "  With GPU:"
    echo "    python run.py --task tsc --agent presslight --world sumo --network sumo4x4 --ngpu 0 --dataset onfly"
    echo ""
    echo "============================================="
}

# ── Main ────────────────────────────────────────────────────────────────────
main() {
    info "Starting LibSignal setup..."
    echo ""

    install_conda
    create_env
    install_pytorch
    install_torch_geometric
    install_sumo
    install_deps
    verify
    print_summary
}

main "$@"
