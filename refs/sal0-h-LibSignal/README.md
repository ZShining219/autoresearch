# LibSignal

[Website](https://darl-libsignal.github.io/)
GitHub Repo stars

OpenAI Gymnasium-compatible environments for **traffic signal control (TSC)** with classical and reinforcement-learning baselines.

**Maintained at:** [sal0-h/LibSignal](https://github.com/sal0-h/LibSignal) — standalone project with Python 3.10+ tooling, SUMO-focused workflows, and team/server setup (`setup.sh`, [docs/team_instructions.pdf](./docs/team_instructions.pdf)).

### Upstream LibSignal (please cite)

This codebase is based on the open-source **[LibSignal](https://github.com/DaRL-LibSignal/LibSignal)** library by the DaRL group ([project site](https://darl-libsignal.github.io/)). We gratefully use their environments, baselines, and simulator integrations; **academic work should cite the original publication** (see [Citation](#citation) below), not only this maintained copy.


|                         |                                                                                                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Original repository** | [https://github.com/DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal)                                                                                     |
| **Paper**               | Mei, H. et al., *Libsignal: an open library for traffic signal control*, Machine Learning (2023). [doi:10.1007/s10994-023-06412-y](https://doi.org/10.1007/s10994-023-06412-y) |


The upstream repository is largely inactive; use **this repo** for installs and day-to-day experiments.

Environments cover single- and multi-intersection networks. Baselines include MaxPressure, fixed-time, SOTL, DQN, PressLight, CoLight, MPLight, and others.

**Simulator focus here:** SUMO (`--world sumo`). CityFlow/OpenEngine paths exist in the codebase from upstream but are not actively tested in this repo.

> **Upstream Docker (optional, not maintained here):** the upstream DaRL group publishes a
> Docker image bundling LibSignal with two sim-to-real projects (UGAT, PromptGAT):
> `docker pull danielda1/ugat:latest`. It is unrelated to this fork's SUMO workflow — use the
> [install steps below](#install) for day-to-day experiments.

# Install

Developed and tested with **SUMO** (`--world sumo`).

## Quick setup (recommended)

```bash
git clone https://github.com/sal0-h/LibSignal.git
cd LibSignal
chmod +x setup.sh
./setup.sh
```

`setup.sh` creates the conda env `traffic` (Python 3.10), installs PyTorch (CUDA if available), SUMO 1.26 (`libsumo` / `traci`), `torch-geometric`, `torch-scatter` (for CoLight), and the Python packages in `requirements.txt`.

On a shared server where system packages are already installed:

```bash
./setup.sh --no-sudo
```

Activate before running experiments:

```bash
conda activate traffic
python run.py --task tsc --agent presslight --world sumo --network sumo1x1 --prefix test
```

## Manual / partial install

If you already have conda and CUDA, you can install pip dependencies after PyTorch and SUMO:

```bash
pip install -r requirements.txt
```

CoLight also needs `torch-scatter` (the setup script installs it via conda-forge). See [docs/team_instructions.pdf](./docs/team_instructions.pdf) for Colab and lab-server workflows.

## Optional: CityFlow

Upstream LibSignal supports `--world cityflow` if [CityFlow](https://github.com/cityflow-project/CityFlow) is installed. We do not test that path here; use SUMO for experiments. A CityFlow ↔ SUMO converter lives in [common/converter.py](./common/converter.py).

## Agents

RL agents are imported automatically from `agent/__init__.py` when their dependencies are
present (e.g. CoLight needs `torch_scatter` + `torch_geometric`). Classical baselines
(`maxpressure`, `fixedtime`, `sotl`) work without those extras.

**Agent status** (registered name → usable via `--agent`):

| Agent | Status | Notes |
|-------|--------|-------|
| `maxpressure`, `fixedtime`, `sotl` | ✅ baseline | No RL deps required. |
| `jev` | ✅ inference | `typesafe/jev-1.13` through OpenRouter's native Decisions API. [Setup and prompt translation](docs/JEV_CONTROLLER.md). |
| `dqn`, `presslight`, `frap`, `mplight`, `magd` | ✅ RL | Standard PyTorch. |
| `ppo_pfrl` | ✅ RL | IPPO via `pfrl`. This is the working PPO. |
| `colight` | ✅ RL | Needs `torch_scatter` (installed by `setup.sh`; not present on the Cloud `.venv`). |
| `maddpg_v2` | ✅ RL | The working MADDPG implementation. |
| `ppo`, `sac`, `maddpg` | ❌ not wired | Registered in their files but **not imported** in `agent/__init__.py`, so they are not in the registry (and `ppo`/`maddpg` reference config keys that are never created). Kept for reference; use `ppo_pfrl` / `maddpg_v2` instead. |

Run `bash scripts/diagnostic.sh` to print the live registry and verify your environment.

# Start

## Run Model Pipeline

Our library has a uniform structure that empowers users to start their experiments with just one click. Users can start an experiment by setting arguments in the run.py file and start with their customized settings. The following part is the arguments provided to customize.

```
python run.py
```

Supporting parameters:

- `--task`: task type to run (default `tsc`).
- `--agent`: agent type — see the [Agents](#agents) table (default `dqn`).
- `--world`: simulator, `sumo` or `cityflow` (use `sumo`; default is `cityflow` from upstream).
- `--network`: network name, maps to `configs/sim/<network>.cfg` (e.g. `sumo1x1`, `sumo4x4`).
- `--prefix`: run name used in the output path.
- `--seed`: seed for the PyTorch/NumPy backend.
- `--ngpu`: GPU id to use; `-1` forces CPU.
- `--interface`: SUMO backend, `libsumo` (fast, default) or `traci` (slower).
- `--delay_type`: delay metric, `apx` (default) or `real`.
- `--thread_num`: worker threads (CityFlow only).
- `--dataset`: dataset handler in training (default `onfly`).

# Documentation

Deep-dive docs live in [`docs/`](./docs/README.md):

- [docs/TRAINING_GUIDE.md](./docs/TRAINING_GUIDE.md) — train/test workflow, config system, ghost-physics baselines.
- [docs/EFFICIENCY_AUDIT.md](./docs/EFFICIENCY_AUDIT.md) — training wall-time audit (GPU vs SUMO vs Python overhead) and speedups.
- [docs/SUMO_NETWORKS.md](./docs/SUMO_NETWORKS.md) — catalogue of available SUMO networks.
- [docs/SUMO_FILE_STRUCTURE_GUIDE.md](./docs/SUMO_FILE_STRUCTURE_GUIDE.md) — SUMO roadnet/flow file structure.
- [docs/TSC_CONFIG_REPORT.md](./docs/TSC_CONFIG_REPORT.md) — code-traced reference for every `configs/tsc/*.yml` parameter.
- [docs/SIGNAL_CONTROL_THEORY.md](./docs/SIGNAL_CONTROL_THEORY.md) — signal-control / NEMA theory + cross-network audit.
- [docs/TECHNICAL_ANALYSIS.md](./docs/TECHNICAL_ANALYSIS.md) — architecture deep-dive.

For the Cursor-Cloud agent environment, see [AGENTS.md](./AGENTS.md).

# Citation

If you use LibSignal (including this maintained repository) in research, **cite the original LibSignal paper and reference the upstream repository**:

- **Paper:** Mei, H., Lei, X., Da, L. et al. Libsignal: an open library for traffic signal control. *Machine Learning* (2023). [https://doi.org/10.1007/s10994-023-06412-y](https://doi.org/10.1007/s10994-023-06412-y)  
- **Code (original):** [https://github.com/DaRL-LibSignal/LibSignal](https://github.com/DaRL-LibSignal/LibSignal)

A short version was also presented at the NeurIPS 2022 Workshop *Reinforcement Learning for Real Life*.

```bibtex
@article{mei2023libsignal,
  title={Libsignal: an open library for traffic signal control},
  author={Mei, Hao and Lei, Xiaoliang and Da, Longchao and Shi, Bin and Wei, Hua},
  journal={Machine Learning},
  pages={1--37},
  year={2023},
  publisher={Springer},
  doi={10.1007/s10994-023-06412-y}
}
```

