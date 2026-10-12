# LibSignal Documentation

Deep-dive documentation for this fork. For install and a quick start, see the
top-level [`README.md`](../README.md); for the Cursor-Cloud agent environment, see
[`AGENTS.md`](../AGENTS.md).

> **Scope reminder:** this fork is developed and tested on **SUMO** (`--world sumo`).
> CityFlow / OpenEngine code paths exist from upstream but are **not tested here**.

## Guides

| Doc | What it covers |
|-----|----------------|
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | How to train and test agents: commands, config system, train-vs-test workflow, ghost-physics baselines. |
| [EFFICIENCY_AUDIT.md](EFFICIENCY_AUDIT.md) | Training wall-time audit: GPU vs SUMO vs Python overhead, measured speedups, profiling scripts. |
| [SUMO_NETWORKS.md](SUMO_NETWORKS.md) | Catalogue of available SUMO networks (sizes, paths, run examples). |
| [SUMO_FILE_STRUCTURE_GUIDE.md](SUMO_FILE_STRUCTURE_GUIDE.md) | Structure of SUMO `roadnet`/`flow` files and how LibSignal reads them. |

## Reference

| Doc | What it covers |
|-----|----------------|
| [AGENT_OBSERVATIONS.md](AGENT_OBSERVATIONS.md) | Why `sumo4x4` creates 32 agents, and exact observation/policy inputs for DQN, IPPO, CoLight, PressLight, MaxPressure. |
| [INPUT_STATE_DESIGN.md](INPUT_STATE_DESIGN.md) | Explicit rich-input variants of DQN, PressLight, and CoLight: features, sensor semantics, configs, and bounded smoke checks. |
| [JEV_CONTROLLER.md](JEV_CONTROLLER.md) | Jev through OpenRouter: Traffic-R1 prompt translation, architecture, configuration, failure behavior, and source documentation. |
| [TSC_CONFIG_REPORT.md](TSC_CONFIG_REPORT.md) | Code-traced reference for every parameter in `configs/tsc/*.yml`, including dead/unused keys. |
| [SIGNAL_CONTROL_THEORY.md](SIGNAL_CONTROL_THEORY.md) | Signal-control theory primer (movements, SUMO signal encoding, NEMA phasing) + a cross-network conflict audit. |
| [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) | Architecture deep-dive: folder roles, state/reward/action definitions, heterogeneous traffic, reaction-time notes. |

## Background & team

| Doc | What it covers |
|-----|----------------|
| [awesome-RL-traffic-signal-control-papers.md](awesome-RL-traffic-signal-control-papers.md) | Curated RL-TSC paper list (inherited from upstream). |
| [team_instructions.pdf](team_instructions.pdf) / [.tex](team_instructions.tex) | Team onboarding / lab-server workflow notes. |
