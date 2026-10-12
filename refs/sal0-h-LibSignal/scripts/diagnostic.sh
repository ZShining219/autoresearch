#!/bin/bash
# LibSignal environment diagnostic.
#
# Consolidates the old colight_test.sh, test_colight_server.sh, and
# server_diagnostic.sh into one script. Checks the Python/SUMO/agent stack and
# probes the CoLight import (the most fragile agent, needs torch_scatter).
#
# Environment: prefers the conda `traffic` env (the canonical dev setup created
# by setup.sh); falls back to a local .venv (the Cursor-Cloud setup); otherwise
# uses whatever Python is already active.
#
# Usage:  bash scripts/diagnostic.sh
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== LibSignal Diagnostic ==="
echo "Date: $(date)"
echo "User: $(whoami)   Host: $(hostname)"
echo ""

# --- Activate an environment (best effort) ------------------------------------
echo "=== Environment activation ==="
if command -v conda &> /dev/null && conda env list 2>/dev/null | grep -qE '^\s*traffic\s'; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate traffic
    echo "Activated conda env: traffic"
elif [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated local .venv"
else
    echo "No conda 'traffic' env or .venv found; using current Python."
fi
echo ""

# --- Python & key packages ----------------------------------------------------
echo "=== Python packages ==="
python --version
python - <<'PY'
import importlib
def check(mod, label=None):
    label = label or mod
    try:
        m = importlib.import_module(mod)
        print(f"{label}: OK", getattr(m, "__version__", ""))
    except Exception as e:
        print(f"{label}: FAILED -> {e}")

import sys
print("executable:", sys.executable)
check("torch")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available(),
          "| GPU count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    pass
check("libsumo")
check("traci")
check("sumolib")
check("torch_geometric")
check("torch_scatter")  # required by CoLight
PY
echo ""

# --- System libraries (SUMO GUI deps) -----------------------------------------
echo "=== System libraries ==="
for lib in libGL.so.1 libglib-2.0.so.0 libgthread-2.0.so.0 libgtk-3.so.0; do
    if ldconfig -p 2>/dev/null | grep -q "$lib"; then echo "$lib: Found"; else echo "$lib: MISSING"; fi
done
echo ""

# --- SUMO ---------------------------------------------------------------------
echo "=== SUMO ==="
if command -v sumo &> /dev/null; then echo "sumo: $(which sumo)"; sumo --version 2>/dev/null | head -1; else echo "sumo binary: not found"; fi
command -v sumo-gui &> /dev/null && echo "sumo-gui: $(which sumo-gui)" || echo "sumo-gui: not found"
echo "SUMO_HOME: ${SUMO_HOME:-<unset>}"
if [ -n "${SUMO_HOME:-}" ] && [ ! -d "$SUMO_HOME" ]; then echo "  WARNING: SUMO_HOME points to a missing directory"; fi
echo ""

# --- Agent registry + CoLight import probe ------------------------------------
echo "=== Agent registry ==="
python - <<'PY'
try:
    import agent  # triggers registration side-effects
    from common.registry import Registry
    names = sorted(Registry.mapping['model_mapping'].keys())
    print("registered models:", names)
    print("colight loadable:", 'colight' in Registry.mapping['model_mapping'])
except Exception as e:
    import traceback; traceback.print_exc()
    print("agent import: FAILED ->", e)

# Direct CoLight probe (its torch_scatter dep is the usual failure point)
try:
    from agent.colight import CoLightAgent
    print("direct CoLight import: SUCCESS")
except Exception as e:
    print(f"direct CoLight import: FAILED -> {type(e).__name__}: {e}")
PY
echo ""
echo "=== Diagnostic complete ==="
