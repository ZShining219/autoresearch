# Agent Count and Observation Inputs

Code-traced answers to [issue #27](https://github.com/sal0-h/LibSignal/issues/27): how many
agents exist per network (especially `sumo4x4` → 32), and what input each of DQN, IPPO,
CoLight, PressLight, and MaxPressure actually consumes.

> **Scope:** SUMO (`--world sumo`) only. Lengths below were measured against the current
> nets; other maps differ by lane/phase count.

## Verdicts (read this first)

1. **32 agents on `sumo4x4` is expected, not a trainer bug.** LibSignal creates **one
   controllable unit per SUMO traffic-light ID**. The `grid4x4` net has **32** TL nodes
   (16 interior grid junctions + 16 fringe portal lights). Docs that say “4×4 = 16
   intersections” describe only the interior grid.
2. **Observation spaces are not matched across agents.** PressLight sees incoming *and*
   outgoing lane counts; DQN/PressLight (default) append a phase one-hot; IPPO and CoLight
   defaults do not; CoLight alone fuses neighbors via a GNN; MaxPressure mostly ignores the
   Gym observation and recomputes movement pressure at action time.

---

## 1. How agents are created

```text
SUMO trafficlight.getIDList()
        │  world/world_sumo.py
        ▼
world.intersections / intersection_ids   (N TLs)
        │  trainer/tsc_trainer.py create_agents()
        ▼
num_agent = len(intersections) / agent.sub_agents
        │
        ├─ typical RL / classical (sub_agents=1):  N agent objects
        └─ CoLight / MPLight (sub_agents=N):       1 agent object, N controllable units
        │
        ▼
environment.TSCEnv asserts len(intersection_ids) == len(agents) * sub_agents
```

Relevant code:

| Step | File | Behavior |
|------|------|----------|
| TL list → intersections | [`world/world_sumo.py`](../world/world_sumo.py) | `intersection_ids = eng.trafficlight.getIDList()`; one `Intersection` per ID |
| Agent factory | [`trainer/tsc_trainer.py`](../trainer/tsc_trainer.py) `create_agents` | `num_agent = len(world.intersections) / agent.sub_agents`; logs `[Device] Moved {len(self.agents)} agent(s) to …` |
| Env check | [`environment.py`](../environment.py) | `n_agents = len(agents) * agents[0].sub_agents`; assert equals `#intersection_ids` |

For `dqn`, `presslight`, `ppo_pfrl`, `maxpressure`, etc., `sub_agents = 1`, so
**agent count = TL count**. CoLight sets `sub_agents = len(world.intersections)` and builds
one object that still controls every TL.

There is **no filter** that drops fringe / portal lights.

---

## 2. Why `sumo4x4` shows 32 agents

| Source | Count | IDs |
|--------|------:|-----|
| Interior 4×4 grid | 16 | `A0`–`A3`, `B0`–`B3`, `C0`–`C3`, `D0`–`D3` |
| Fringe portals | 16 | `top0–3`, `bottom0–3`, `left0–3`, `right0–3` |
| **Total TLs / agents** | **32** | all nodes in [`data/raw_data/grid4x4/grid4x4.nod.xml`](../data/raw_data/grid4x4/grid4x4.nod.xml) are `type="traffic_light_right_on_red"` |

Smoke log (CPU):

```text
[Device] Moved 32 agent(s) to cpu
```

Interior junctions (e.g. `A0`) have full multi-phase logics (8 green phases after LibSignal’s
phase filter). Fringe logics are trivial (essentially always-green portals: measured **1**
usable phase, **1** incoming lane). They still receive an agent because they appear in
`trafficlight.getIDList()`.

**Interpretation:** feature of “1 agent per TL” + this network’s 16 fringe TLs — not a
double-count in `create_agents`. To get a 16-agent “true 4×4” setup you would need to filter
fringe IDs or change the net; that is out of scope of this document.

---

## 3. Shared observation pipeline

```text
world.get_info("lane_count") ──► LaneVehicleGenerator ──► agent.get_ob()
                                                              │
env.reset / env.step ◄────────────────────────────────────────┘  (obs only)

trainer ──► agent.get_phase() ──► passed separately into get_action / remember
```

Important details:

- **Env observation** ([`environment.py`](../environment.py)): `obs = agent.get_ob()` only.
  Phase is **not** part of the Gym obs list.
- **Phase in the policy:** some agents concatenate phase inside `get_action` / replay
  batching when `model.phase: True` (see configs below).
- **Feature builder:** [`generator/lane_vehicle.py`](../generator/lane_vehicle.py)
  (`LaneVehicleGenerator`).
  - `in_only=True` → incoming roads only; default `in_only=False` → incoming + outgoing.
  - `average=None` → one scalar per lane (vehicle count for `lane_count`).
  - If the raw length is 2 or 3, the generator **pads to 4**.
- **Lane order:** roads as stored on the intersection; within a road, lanes sorted by trailing
  lane index.
- **Phase generator:** [`generator/intersection_phase.py`](../generator/intersection_phase.py)
  returns the current green-phase index.

Available world features that these five agents mostly **do not** put in the state vector
(used for reward/metrics instead): `lane_waiting_count`, `lane_delay`,
`lane_waiting_time_count`, `lane_pressure`, scalar `pressure`.

Partial-observability corruption ([`docs/PARTIAL_OBSERVABILITY.md`](PARTIAL_OBSERVABILITY.md))
applies to `lane_count` (and derived pressures) upstream of every agent the same way.

---

## 4. Measured vector lengths

Lengths from `LaneVehicleGenerator.ob_length` and `#phases` on the live SUMO world:

| Network | Role | Count | `in_only` len | in+out len | Green phases |
|---------|------|------:|--------------:|-----------:|-------------:|
| `sumo1x1` | single TL | 1 | 8 | 16 | 4 |
| `sumo4x4` | interior (`A0`…) | 16 | 12 | 24 | 8 |
| `sumo4x4` | fringe (`top0`…) | 16 | 1 | 4 | 1 |

Effective **policy** input size ≈ obs length + (phase one-hot size if enabled):

| Agent | sumo1x1 policy dim | sumo4x4 interior policy dim |
|-------|-------------------:|----------------------------:|
| DQN | 8 + 4 = **12** | 12 + 8 = **20** |
| IPPO (`ppo_pfrl`) | **8** | **12** |
| CoLight | **8** (local; + GNN) | **12** padded across nodes (local; + GNN) |
| PressLight | 16 + 4 = **20** | 24 + 8 = **32** |
| MaxPressure | `get_ob` len 8 / 12; action ignores it | same |

CoLight zero-pads each node’s local vector to `max` incoming length across all TLs (on
`sumo4x4`, fringe length-1 vectors pad to 12).

---

## 5. Per-agent inputs

### DQN — [`agent/dqn.py`](../agent/dqn.py), [`configs/tsc/dqn.yml`](../configs/tsc/dqn.yml)

| Piece | Setting |
|-------|---------|
| `get_ob()` | `lane_count`, `in_only=True` → shape `(1, L)` |
| Config | `phase: True`, `one_hot: True` |
| Policy input | `[incoming lane counts ‖ one-hot phase]` |
| Reward | mean incoming `lane_waiting_count` (negated, ×12) |
| Neighbors | none |

```text
[ n_lane_0, …, n_lane_{L-1},  phase_onehot_0, …, phase_onehot_{P-1} ]
```

### IPPO — [`agent/ppo_pfrl.py`](../agent/ppo_pfrl.py) (`--agent ppo_pfrl`), [`configs/tsc/ppo_pfrl.yml`](../configs/tsc/ppo_pfrl.yml)

Generators match DQN (`lane_count`, `in_only=True`). The agent reads
`model_mapping['setting'].param['phase']` / `one_hot`.

**Config caveat:** `ppo_pfrl.yml` puts `phase: True` / `one_hot: True` under a top-level
`traffic:` block that this agent **never reads**. Effective flags come from
[`configs/tsc/base.yml`](../configs/tsc/base.yml): **`phase: False`, `one_hot: False`**.

So default IPPO policy input = **incoming `lane_count` only**. The trainer still fetches
phase and passes it to `get_action`, but it is not concatenated when `self.phase` is false.

(Orphan [`agent/ppo.py`](../agent/ppo.py) is not registered for `--agent`.)

### CoLight — [`agent/colight.py`](../agent/colight.py), [`configs/tsc/colight.yml`](../configs/tsc/colight.yml)

| Piece | Setting |
|-------|---------|
| Local feature | same as DQN: `lane_count`, `in_only=True` per intersection |
| `get_ob()` | stack of N node vectors, each `/ vehicle_max` (default 1), zero-padded to max L |
| Config | `phase: False` (phase unused in `get_action`; comment in code) |
| Neighbors | **not** in the vector — multi-head attention over roadnet `sparse_adj` |

Same local feature type as DQN; richer **effective** context on multi-intersection nets via
the graph. One agent object (`sub_agents = N`) still covers all 32 TLs on `sumo4x4`.

### PressLight — [`agent/presslight.py`](../agent/presslight.py), [`configs/tsc/presslight.yml`](../configs/tsc/presslight.yml)

| Piece | Setting |
|-------|---------|
| `get_ob()` | `lane_count`, **`in_only` default False** → incoming **and** outgoing lanes |
| Config | `phase: True`, `one_hot: True` |
| Policy input | `[in+out lane counts ‖ one-hot phase]` |
| Reward | scalar intersection `pressure` (Σ in − Σ out), not part of the state vector |

Richest **local lane-count** observation among these RL agents. Comments in the agent
(`ob_length` 32) match `sumo4x4` interior: 24 lanes + 8-phase one-hot.

### MaxPressure — [`agent/maxpressure.py`](../agent/maxpressure.py), [`configs/tsc/maxpressure.yml`](../configs/tsc/maxpressure.yml)

| Piece | Setting |
|-------|---------|
| `get_ob()` | incoming `lane_count` (env / logging interface) |
| `get_action(ob, phase, …)` | **ignores `ob` and `phase` for scoring** |
| Decision | for each phase, `Σ (count[start] − count[end])` over `phase_available_lanelinks`; pick max |
| Hold | if `current_phase_time < t_min` (config default 10), keep current phase |

Different paradigm: classical heuristic using movement-level pressure from live
`world.get_info("lane_count")`, not an NN over the Gym observation vector.

---

## 6. Who has more information?

```text
Richest local counts ──► PressLight (in+out lane_count + phase one-hot)
                         MaxPressure action (per-movement pressure; ignores NN-style obs)
Multi-agent context ───► CoLight (same local counts as DQN + neighbor GNN)
Middle ────────────────► DQN (in-only lane_count + phase one-hot)
Leanest default ───────► IPPO (in-only lane_count, no phase in policy input)
```

| Dimension | DQN | IPPO | CoLight | PressLight | MaxPressure |
|-----------|:---:|:----:|:-------:|:----------:|:-----------:|
| Incoming counts in state | yes | yes | yes | yes | yes (`get_ob`) |
| Outgoing counts in state | no | no | no | **yes** | via lanelinks in **action** |
| Phase in policy input | **yes** (OH) | **no** | no | **yes** (OH) | hold timer only |
| Neighbor intersections | no | no | **yes (GNN)** | no | no |
| Queue / delay / wait in state | no | no | no | no | no |
| Explicit pressure in state | no | no | no | no (reward only) | **yes (decision)** |

These baselines are **not** a fair matched-observation comparison. Prefer PressLight when
comparing to MaxPressure’s in−out structure; prefer DQN when comparing phase-aware local
RL; treat CoLight as the neighbor-sharing variant; treat default IPPO as phase-ablated
local RL unless you move `phase`/`one_hot` under `model:` in its YAML.

---

## 7. Related docs

| Doc | Relation |
|-----|----------|
| [`SUMO_NETWORKS.md`](SUMO_NETWORKS.md) | Network catalogue (incl. `sumo4x4` TL counts) |
| [`TECHNICAL_ANALYSIS.md`](TECHNICAL_ANALYSIS.md) | Broader architecture / state-reward overview |
| [`TSC_CONFIG_REPORT.md`](TSC_CONFIG_REPORT.md) | Every `configs/tsc/*.yml` key, including dead keys |
| [`PARTIAL_OBSERVABILITY.md`](PARTIAL_OBSERVABILITY.md) | How `lane_count` can be corrupted before agents read it |
| [`SIGNAL_CONTROL_THEORY.md`](SIGNAL_CONTROL_THEORY.md) | Phase / movement semantics on `grid4x4` |
