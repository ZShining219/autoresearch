# AGENTS.md

LibSignal — OpenAI Gymnasium-compatible library for traffic signal control (TSC) with
classical and RL baselines. It is a single Python CLI tool (entry point `run.py`); there is
no web server, API, frontend, or database. See `README.md` and `docs/TRAINING_GUIDE.md` for usage.

## Cursor Cloud specific instructions

### Environment
- The dev environment is a Python 3.12 virtualenv at `.venv/` (the repo's `setup.sh` uses a
  conda env named `traffic`, but on Cursor Cloud we use a plain venv instead — it is lighter
  and more reliable). The venv is created/refreshed on startup by the `install` command in
  `.cursor/environment.json` (committed so every Cloud Agent on this repo gets it, regardless
  of personal/team snapshot scope).
- Always work inside the venv: `source .venv/bin/activate`.
- There is **no GPU** on Cloud VMs — always run with `--ngpu -1` (CPU). The code prints
  `[Device] No CUDA GPU detected, using CPU`.

### SUMO_HOME (important gotcha)
- `world/world_sumo.py` calls `sys.exit('No SUMO in environment path')` if `SUMO_HOME` is unset.
- Activating the venv (`source .venv/bin/activate`) auto-exports `SUMO_HOME` to the bundled
  `eclipse-sumo` wheel dir (a line appended to `.venv/bin/activate`). If you ever run the venv
  python without sourcing activate, set it manually:
  `export SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')"`.

### CoLight agent
- The `colight` agent needs `torch_scatter`, which is **not installed** (pip build is heavy/fragile
  and it requires no extra system libs to skip). All other agents work. `agent/__init__.py` imports
  CoLight lazily, so you will see a harmless
  `Warning: Failed to import CoLightAgent: No module named 'torch_scatter'` on every run.
  To use CoLight, install it against the current torch:
  `pip install torch_scatter -f https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)').html`.

### Run / test (always activate the venv first, and pass `--ngpu -1`)
- Lint: no linter is configured in the repo; `python -m compileall agent world trainer task common utils generator dataset run.py environment.py` is the sanity check used here.
- Hello-world / smoke test (instant, no training): `python run.py --agent maxpressure --world sumo --network sumo1x1 --seed 42 --ngpu -1`.
- RL training (slow on CPU): `python run.py --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu -1`. Defaults to 200 episodes with no CLI flag to shorten it; time-box it if you only need to confirm the loop runs (q_loss is logged and travel time drops over episodes).
- Outputs go to `data/output_data/tsc/<world>_<agent>_<prefix>/` (these are runtime artifacts — do not commit them).

### Notes
- Use `--world sumo`; CityFlow/OpenEngine paths exist from upstream but are not tested in this fork.
- `--interface libsumo` (default, fast) works; `--interface traci` is the slower fallback.
