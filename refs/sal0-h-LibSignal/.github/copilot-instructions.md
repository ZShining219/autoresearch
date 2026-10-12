# LibSignal Copilot Instructions

## Project Overview
LibSignal is a reinforcement learning research framework for traffic signal control (TSC). The `world/` layer wraps three simulator backends from upstream (SUMO, CityFlow, OpenEngine), but **this fork develops and tests only the SUMO path (`--world sumo`)** — CityFlow/OpenEngine are retained from upstream and are not validated here. It provides configurable road networks, multiple baseline agents, and a modular architecture for training and evaluating TSC policies.

## Architecture Overview

### Core Components (Understand via run.py flow)
- **Runner**: Entry point that orchestrates config building and component instantiation (see `run.py`)
- **World**: Simulator abstraction layer in `world/` (SUMO/CityFlow/OpenEngine implementations)
- **Agent**: Policy implementations in `agent/` (RL agents, baseline algorithms, fixed-time control)
- **Environment**: OpenAI Gym wrapper (`environment.py`) connecting agents to worlds
- **Trainer**: Training/testing loop in `trainer/` (creates world, environment, handles checkpoints)
- **Task**: High-level execution orchestrator (`task/task.py`) that coordinates trainer

### Data Flow: How Components Connect
1. `run.py` parses args → builds config from `configs/` files
2. Config registered via `common/interface.py` → Registry singleton
3. Trainer instantiated with registered config → creates World + Agents
4. Trainer creates TSCEnv wrapping world and agents
5. Task calls trainer.train() or trainer.test()
6. Agent.step() returns observations/rewards; World updates simulator state

### Configuration System
- **CLI args**: `--task`, `--agent`, `--world`, `--network` define execution
- **Config files**: YAML in `configs/tsc/` and `configs/sim/` (network-specific CFG files)
- **Registry pattern**: Decorator-based registration (see `@Registry.register_model()`) enables pluggable components
- Example: Agent type `dqn` maps to DQN class via `@Registry.register_model('dqn')` decorator

## Key Patterns & Conventions

### Agent Implementation Pattern
All agents inherit `BaseAgent` and implement:
- `get_ob()`: Return observation (network-specific state features from lanes, queues, phases)
- `get_reward()`: Return scalar reward (typically based on queue length, wait time, or pressure)
- `get_action()`: Return phase index or action from observation
- `sub_agents`: Integer controlling agent granularity (1=single intersection, >1=sub-agent splits)

Example: `agent/dqn.py` (RL-based) vs `agent/fixedtime.py` (heuristic baseline)

### World Interface Pattern
Worlds implement simulator-specific logic but expose unified interface:
- `step(actions)`: Apply agent actions; simulator advances one timestep
- `reset()`: Reload network/traffic; reset metrics
- `get_lane_vehicle_state()`, `get_intersection_phases()`: State queries
- Each world manages `Intersection` objects with phase transitions and vehicle tracking

### Multi-Agent Coordination
- Single agent with `sub_agents > 1`: One agent controls multiple intersections via sub-agents
- Multiple agents: List of agents, each controls one/few intersections
- Observation/reward returned as list matching agent count (see `environment.py` step/reset)

## Developer Workflows

### Running Experiments
```bash
python run.py --task tsc --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu -1
```
Common variations:
- `--agent`: baselines `maxpressure`, `fixedtime`, `sotl`; RL `dqn`, `presslight`, `frap`, `mplight`, `magd`, `colight`, `ppo_pfrl`, `maddpg_v2`. (Note: `ppo`, `sac`, `maddpg` exist as files but are **not** wired into `agent/__init__.py` — use `ppo_pfrl` / `maddpg_v2`.)
- `--world`: `sumo` (supported); `cityflow` exists from upstream but is untested here.
- `--network`: sumo1x1, sumo4x4, grid4x4, sumo7x28, etc. (maps to configs/sim/{network}.cfg)
- `--ngpu`: `-1` forces CPU.

### Adding New Agents
1. Create `agent/my_agent.py` inheriting `BaseAgent`
2. Implement `get_ob()`, `get_reward()`, `get_action()`
3. Register via `@Registry.register_model('my_agent_name')` decorator
4. **Import it in `agent/__init__.py`** (otherwise the decorator never runs and the agent won't be in the registry)
5. Add YAML config to `configs/tsc/my_agent_name.yml`
6. Run: `python run.py --agent my_agent_name`

### Debugging Tips
- Set `--debug True` for verbose logging
- Check `Registry.mapping` in common/registry.py to verify component registration
- World configs in `configs/sim/` define network topology and traffic (roadnet paths)
- Agent configs in `configs/tsc/` define hyperparameters loaded by agent __init__

## Critical Integration Points

### Runner → Config → Registry
`run.py` line ~40: `interface.Command_Setting_Interface(self.config)` registers config globally via Registry. Components read config later via:
```python
Registry.mapping['command_mapping']['setting'].param['network']  # network name
Registry.mapping['model_mapping']['setting'].param['train_model']  # train/test flags
```

### Trainer → World → Intersection
Trainer creates world (e.g., `WorldSUMO`) which spawns `Intersection` objects per node. Agents query intersections for state and apply phase actions. Watch state updates in `world_sumo.py` step() method.

### Environment → Gym Compatibility
TSCEnv.step() returns `(obs_list, rewards_list, dones_list, infos_dict)` matching OpenAI Gym. Multi-agent setup returns lists; check environment.py for shape consistency with agent count.

## Project-Specific Gotchas
- **Phase encoding**: Different simulators (SUMO/CityFlow) use different phase numbering; conversion calibrated but watch for mismatches
- **Vehicle state features**: Queues and wait time computed differently per simulator; check `generator/lane_vehicle.py` for state feature definitions
- **SUMO environment variable**: Must set `SUMO_HOME` or world_sumo.py fails at import
- **Multi-agent agent lists**: Trainer expects list of agents; each agent's sub_agents property determines actual control dimensionality
- **Metrics**: Loaded by trainer post-run; check `common/metrics.py` for available metrics (average queue, wait time)

## File Reference Guide
- Entry: [run.py](run.py) (args parsing, Runner instantiation)
- Config loading: [common/interface.py](common/interface.py) (Registry population)
- Agent base: [agent/base.py](agent/base.py) (get_ob/get_reward/get_action contract)
- Example RL agent: [agent/dqn.py](agent/dqn.py) (policy network, exploration)
- Example baseline: [agent/maxpressure.py](agent/maxpressure.py) (heuristic algorithm)
- Simulator interface: [world/world_sumo.py](world/world_sumo.py) and [world/world_cityflow.py](world/world_cityflow.py)
- Environment wrapper: [environment.py](environment.py) (Gym-compatible step/reset)
- Trainer: [trainer/base_trainer.py](trainer/base_trainer.py) and [trainer/tsc_trainer.py](trainer/tsc_trainer.py) (train/test loops)
- Task execution: [task/task.py](task/task.py) (TSCTask orchestration)
- State features: [generator/lane_vehicle.py](generator/lane_vehicle.py) (queue/delay computation)
