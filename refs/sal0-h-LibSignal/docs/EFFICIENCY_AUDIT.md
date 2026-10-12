# Training Efficiency Audit

Deep wall-clock audit of LibSignal TSC training (SUMO + libsumo). Measured on
Cursor Cloud (4×CPU, torch 2.13 CPU-only, no GPU). Date of measurements: 2026-07-13.

## Verdict

**GPU will not help much on the default DQN / classical baselines.** Training wall
time is dominated by the **Python↔libsumo observation loop**, not neural-net
compute. SUMO’s `simulationStep` is a large but not always majority share; a lot
of time is spent in *our* wrappers around SUMO (observe, unused trajectory
tracking, per-agent generators).

After the optimizations in this change set, episode wall time improved by roughly:

| Network | Intersections | Before (s) | After (s) | Speedup |
|---------|---------------|------------|-----------|---------|
| sumo1x1 | 1 | 1.90 | 1.46 | **1.30×** |
| cologne3 | 3 | 3.14 | 2.01 | **1.57×** |
| sumo4x4 | 32 | 10.41 | 6.08 | **1.71×** |
| DQN sumo1x1 (2 ep, no eval) | 1 | 4.69 | 3.63 | **1.29×** |

MaxPressure `sumo1x1` ATT / rewards / queue / delay / throughput were unchanged
at seed 42 (`ATT=39.0070`), so the speedups are behavior-preserving for the
default `--delay_type apx` path.

## Where the time goes (before optimizations)

Exclusive-ish leaf costs inside one MaxPressure episode (3600 sim seconds,
`action_interval=10`):

### sumo1x1 (1 intersection, ~1.9 s wall)

| Component | Seconds | Notes |
|-----------|---------|-------|
| `simulationStep` | 0.26 | Real SUMO physics (untouchable) |
| `Intersection.observe` | 0.15 | Per-lane / per-vehicle libsumo queries |
| `agent.get_reward` | 0.15 | Generator `np.append` churn every step |
| `get_vehicle_trajectory` | 0.12 | **Unused** by DQN/maxpressure |
| `agent.get_ob` | 0.05 | Intermediate obs discarded |
| Agent `get_action` | ~0.002 | Negligible |

### sumo4x4 (32 intersections, ~10.4 s wall)

| Component | Seconds | Notes |
|-----------|---------|-------|
| `agent.get_reward` | **2.55** | 32 agents × 3600 steps |
| `observe` | 1.72 | Scales with intersections × vehicles |
| `simulationStep` | 1.16 | SUMO physics |
| `get_vehicle_trajectory` | 0.95 | Scans **all 832 lanes** every step |
| `agent.get_ob` | 0.91 | 9/10 results thrown away |

**Takeaway:** On multi-intersection maps, Python agent/generator overhead can
exceed SUMO physics. “It’s all SUMO” is only half true.

## GPU vs CPU (DQN)

`DQNNet` is a tiny MLP: `input → 20 → 20 → n_actions` (~1k parameters).

Standalone CPU microbench during this audit:

| Call | Mean (CPU) |
|------|------------|
| `train()` batch=64 | **0.65 ms** |
| `get_action` forward batch=1 | **0.019 ms** |

At 360 decisions/episode after `learning_start`, RL compute ≈ **0.24 s/episode**
vs ~1 s of env time on sumo1x1 (~24% when training every decision). A GPU cannot
saturate on ~700k FLOPs/step; kernel-launch overhead eats the gain. Keep
`--ngpu -1` on Cloud / small maps.

GPU would matter for large graph models (e.g. CoLight) or huge batches — not this DQN.

## What we cannot change

- **SUMO / libsumo physics** (`simulationStep`) — fixed cost per sim second.
- Need for 3600 sim steps/episode at 1 s resolution (scenario length).
- Yellow / phase logic that requires stepping the simulator every second.

## What we fixed (this PR)

1. **Gate `get_vehicle_trajectory`** (`world_sumo.py`)  
   It ran *every* step even though no shipped agent subscribes to it. Now it
   runs only if `"vehicle_trajectory"` is subscribed or
   `world.update_vehicle_trajectory=True` (set automatically when
   `--delay_type real`).

2. **Faster `LaneVehicleGenerator.generate`** (`generator/lane_vehicle.py`)  
   Replaced repeated `np.append` with list accumulation + one array conversion.
   Preserves original averaging semantics (`average="all"` = mean of per-road means).

3. **Skip intermediate observations** (`environment.py`, `trainer/tsc_trainer.py`)  
   Inside each `action_interval` block, only the **last** `get_ob()` is kept for
   the replay / next-state. Intermediate `get_ob` calls were pure waste. Rewards
   are still collected every sim step so interval averaging is unchanged.

4. **DQN `train()` one fewer forward** (`agent/dqn.py`)  
   Previously forwarded `b_t` twice (eval then train). Now one forward + scatter
   into a detached target clone. Equivalent for `DQNNet` (no dropout/batchnorm).

## Config levers (documentation only — no YAML defaults changed in this PR)

| Lever | Effect |
|-------|--------|
| `trainer.test_when_train: False` | Removes a full extra rollout **per training episode** (~2× less sim for DQN defaults). |
| Fewer `episodes` / `steps` | Linear wall-time cut. |
| `action_interval` | Lower → more decisions + more RL updates; does **not** reduce sim steps. |
| `--interface libsumo` | Already default; `traci` is much slower. |
| `--delay_type apx` | Default; `real` re-enables per-step trajectories. |

## Remaining opportunities (not done)

| Idea | Risk | Expected impact |
|------|------|-----------------|
| Collect reward only on decision boundaries (no 10-step mean) | Behavior change | Large on multi-agent |
| Faster `observe` / `_get_vehicles` without `getNextTLS` | Possible obs change | Medium |
| Cache `lane.getMaxSpeed` in `get_lane_delay` | Low | Small |
| Don’t subscribe `lane_delay` every step; compute at decision rate | Low–medium | Small–medium |
| Vectorized / multi-process envs | Large engineering | Only if you need many seeds/episodes in parallel |
| Parallel SUMO instances for population-based training | Engineering | Throughput, not single-episode latency |
