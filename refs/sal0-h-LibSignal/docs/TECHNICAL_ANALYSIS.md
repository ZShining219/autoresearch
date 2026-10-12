# LibSignal Technical Analysis

Deep dive into LibSignal's architecture, state/reward/action definitions, and extensibility for advanced traffic scenarios.

> **Scope note:** the `world/` layer contains three simulator backends from upstream
> (`world_sumo.py`, `world_cityflow.py`, `world_openengine.py`), and this document describes
> all three where relevant. In **this fork, only the SUMO path (`--world sumo`) is tested and
> supported**; CityFlow/OpenEngine references are retained for architectural completeness and
> upstream compatibility, not as validated workflows.

---

## 1. Folder Structure & Component Roles

### Core Directories

#### `agent/`
**Role**: Policy implementations (RL and baseline algorithms)
- **Base classes**: `BaseAgent` (all agents), `RLAgent` (RL-specific)
- **RL agents (wired & usable)**: `dqn.py`, `presslight.py`, `frap.py`, `mplight.py`, `magd.py`, `colight.py`, `ppo_pfrl.py` (IPPO), `maddpg_v2.py`
- **Baselines**: `maxpressure.py` (queue-based pressure), `sotl.py` (self-optimizing), `fixedtime.py` (static timing)
- **Orphan / superseded files (present but NOT wired into `agent/__init__.py`)** — kept for
  reference/history, not usable via `--agent`:
  - `ppo.py` (`@register('ppo')`), `sac.py` (`@register('sac')`), `maddpg.py` (`@register('maddpg')`)
    — decorators never run (not imported), so these names are **absent from the registry**;
    `ppo.py`/`maddpg.py` also read config keys (`model_setting`/`traffic_setting`) that are never
    created. Use `ppo_pfrl` / `maddpg_v2` instead.
  - `colight_pytorch_agent.py`, `dqn_torch_agent.py` — earlier implementations superseded by
    `colight.py` / `dqn.py` (zero references).
  - `maddpg_agent.py` + `maddpg_agent_util.py` — legacy MADDPG helper cluster (only referenced by
    each other), superseded by `maddpg_v2.py`.
- **Key methods each agent must implement**:
  - `get_ob()`: Extract observation from world state
  - `get_reward()`: Compute reward signal
  - `get_action(ob, phase, test=False)`: Policy decision (explores if `test=False`)
  - `train()`: Update model parameters (RL agents only)
  - `save_model(e)` / `load_model(e)`: Checkpoint persistence

#### `world/`
**Role**: Simulator abstraction layer (SUMO, CityFlow, OpenEngine)
- **Files**: `world_sumo.py`, `world_cityflow.py`, `world_openengine.py`
- **Key classes**:
  - `World`: Top-level simulator interface with `step(actions)`, `reset()`, `get_info(fn)`
  - `Intersection`: Per-junction state management (phases, lanes, vehicle tracking)
- **Subscription system**: Agents subscribe to state features (e.g., `lane_count`, `lane_waiting_count`) via `world.subscribe(fns)`; world collects them during `step()`
- **State queries**: `get_lane_vehicle_count()`, `get_lane_waiting_vehicle_count()`, `get_lane_waiting_time_count()`, `get_lane_delay()`, `get_lane_pressure()`

#### `generator/`
**Role**: State/reward feature extractors from world observations
- **`lane_vehicle.py` (`LaneVehicleGenerator`)**: Extract lane-level stats
  - Supported features: `lane_count`, `lane_waiting_count`, `lane_waiting_time_count`, `lane_delay`, `lane_pressure`
  - Aggregation modes: `in_only=True` (incoming lanes only), `average='all'/'road'/None`
  - Used for both **state** (observations) and **reward** (negative waiting count)
- **`intersection_phase.py` (`IntersectionPhaseGenerator`)**: Current phase info
- **`intersection_vehicle.py` (`IntersectionVehicleGenerator`)**: Vehicle trajectory-based features (used by IntelliLight-style agents)

**Critical Design**: Generators decouple state/reward definitions from agent logic. Each agent instantiates generators with specific parameters in `__init__()`.

#### `trainer/`
**Role**: Training/testing orchestration
- **`base_trainer.py`**: Abstract class defining `create_world()`, `create_agents()`, `create_env()`, `train()`, `test()`
- **`tsc_trainer.py`**: Traffic signal control trainer
  - **Training loop**: Runs episodes, collects transitions, calls `agent.train()` after `learning_start` steps
  - **Checkpointing**: Saves models every `save_rate` episodes to `{output_dir}/model/{episode}_{agent_rank}.pt`
  - **Testing**: Evaluates policy without exploration (`test=True` in `get_action()`)

#### `environment.py`
**Role**: OpenAI Gym wrapper (`TSCEnv`)
- **Interface**: `step(actions)` → `(obs, rewards, dones, infos)`, `reset()` → `obs`
- **Multi-agent handling**: Returns lists of obs/rewards matching agent count (each agent may control multiple intersections via `sub_agents`)

#### `configs/`
**Role**: Hierarchical YAML configuration
- **`tsc/base.yml`**: Default settings (episodes, steps, buffer size, learning rate, etc.)
- **`tsc/{agent}.yml`**: Agent-specific overrides (e.g., `dqn.yml` sets `train_model: True`, `epsilon: 0.1`)
- **`sim/{network}.cfg`**: Network topology (roadnet file paths, traffic flow files, simulator settings)
- **Inheritance**: Agent configs include `base.yml`, overriding specific parameters

#### `common/`
**Role**: Shared utilities
- **`registry.py`**: Global decorator-based registration system for agents, trainers, worlds, tasks
- **`interface.py`**: Config registration into Registry (called by `Runner.config_registry()`)
- **`converter.py`**: SUMO ↔ CityFlow network/traffic conversion tools
- **`metrics.py`**: Evaluation metrics (queue, delay, throughput, travel time)

#### `data/raw_data/`
**Role**: Network definitions and traffic flows
- **Structure**: `{network_name}/roadnet.json` (CityFlow) or `.net.xml` (SUMO), `flow.json` or `.rou.xml` (traffic)
- **Examples**: `hangzhou_1x1_bc-tyc_18041607_1h/`, `grid4x4/`, `cologne1/`

#### `task/`
**Role**: High-level execution orchestrator
- **`task.py`**: `TSCTask` calls `trainer.train()` and/or `trainer.test()` based on config flags (`train_model`, `test_model`)

#### `utils/`
**Role**: Logging and output path management
- **`logger.py`**: Sets up file/console loggers, generates output directories (`data/output_data/tsc/{experiment_name}/`)

---

## 2. State, Reward, and Action Definitions

### Where They're Defined

#### **State (Observation)**
Defined **per-agent** in `agent/{agent}.py` via `LaneVehicleGenerator` instantiation:

**Example (DQN):**
```python
# agent/dqn.py line 40
self.ob_generator = LaneVehicleGenerator(
    self.world, self.inter, 
    ['lane_count'],         # Feature: vehicle count per lane
    in_only=True,           # Only incoming lanes
    average=None            # No aggregation (per-lane values)
)
```

**Available features** (in `generator/lane_vehicle.py`):
- `lane_count`: Number of vehicles on each lane
- `lane_waiting_count`: Vehicles with speed < 0.1 m/s
- `lane_waiting_time_count`: Cumulative waiting time on lane
- `lane_delay`: `1 - avg_speed / speed_limit`
- `lane_pressure`: `in_lane_vehicles - out_lane_vehicles`

**Observation shape**:
- Base: `[num_incoming_lanes * num_features]`
- With phase: Concatenates current phase (one-hot or scalar)
- Example: 12 incoming lanes, 1 feature → `[12]` or `[12 + num_phases]` if phase included

#### **Reward**
Also defined **per-agent** via `LaneVehicleGenerator`:

**Example (DQN):**
```python
# agent/dqn.py line 44
self.reward_generator = LaneVehicleGenerator(
    self.world, self.inter,
    ["lane_waiting_count"],  # Feature: waiting vehicles
    in_only=True,
    average='all',           # Average across all lanes → scalar
    negative=True            # Return negative (minimize waiting)
)
```

**Common reward choices**:
- **Negative queue**: `-sum(lane_waiting_count)` (minimize waiting vehicles)
- **Negative delay**: `-avg(lane_delay)` (minimize travel time loss)
- **Pressure**: `-abs(in_lanes - out_lanes)` (balance flow)
- **Throughput**: `+vehicles_passed` (maximize flow)

**Called in**:
```python
# agent/dqn.py line 108
def get_reward(self):
    return self.reward_generator.generate()
```

#### **Action**
- **Action space**: Discrete phase selection (0 to `num_phases - 1`)
- **Defined by**: Intersection's phase configuration in roadnet (SUMO `.net.xml` or CityFlow `roadnet.json`)
- **Agent action**: Index into available phases
- **World execution**: `world.step(actions)` applies phase to traffic light, simulates `action_interval` steps

**Phase transitions**:
- Yellow phase inserted automatically between green phase changes (duration: `yellow_length` from config, default 5s)
- Virtual phase tracking: `Intersection.virtual_phase` (treats current yellow same as previous green)

### Fairness of Comparing Algorithms with Different States

**Key Question**: Is it fair to compare DQN using `lane_count` vs. PPO using `lane_waiting_count + lane_delay`?

**Answer**: **No, not directly fair.** LibSignal allows arbitrary state definitions per agent, which violates controlled comparison principles:

**Problems**:
1. **Information asymmetry**: Richer state (more features) gives advantage
2. **Observation dimensionality**: Affects network capacity requirements
3. **Reward coupling**: State features often overlap with reward (e.g., using `lane_waiting_count` for both state and reward can cause reward hacking)

**Best Practices for Fair Comparison**:
1. **Fix state/reward across agents**: Use identical `ob_generator` and `reward_generator` parameters
2. **Document differences**: If testing state design, explicitly report it
3. **Ablation studies**: Compare same agent with different states to isolate impact
4. **Standardize**: Many papers use `[lane_count, current_phase]` as state baseline

**Example Fair Setup** (modify agent code):
```python
# Standardized state for all RL agents
ob_generator = LaneVehicleGenerator(world, inter, ['lane_count'], in_only=True, average=None)
reward_generator = LaneVehicleGenerator(world, inter, ["lane_waiting_count"], in_only=True, average='all', negative=True)
```

**LibSignal's Flexibility**: Useful for **state representation research** but requires discipline for **algorithm benchmarking**.

---

## 3. Heterogeneous Traffic Support (Bikes, Trucks, etc.)

### Current State

**Vehicle parameters defined in flow files**:
- **CityFlow** (`flow.json`): Each flow entry has `vehicle` object:
  ```json
  "vehicle": {
    "length": 5.0,
    "width": 2.0,
    "maxPosAcc": 2.0,
    "maxNegAcc": 4.5,
    "minGap": 2.5,
    "maxSpeed": 11.11,
    "headwayTime": 2.0
  }
  ```
- **SUMO** (`.rou.xml`): Uses `<vType>` definitions:
  ```xml
  <vType id="car" length="5.0" width="2.0" accel="2.6" decel="4.5" minGap="2.5"/>
  <vType id="truck" length="12.0" width="2.5" accel="1.5" decel="3.0" minGap="5.0"/>
  <vehicle id="veh_0" type="truck" depart="10" route="route_1"/>
  ```

### Heterogeneity Support Analysis

#### **SUMO: Full Support ✅**
- Native `vType` system allows arbitrary vehicle classes
- Define multiple types in `<additional>` file or inline in `.rou.xml`
- Assign types per vehicle in routes
- **Car-following models**: IDM, Krauss, W99, etc. (different acceleration profiles)
- **Lane change models**: LC2013, SL2015 (bikes can filter, trucks lane-restricted)
- **Access permissions**: `<lane allow="bike truck" disallow="passenger"/>`

**Example (trucks and bikes)**:
```xml
<vTypes>
  <vType id="car" length="5.0" width="2.0" accel="2.6" decel="4.5" vClass="passenger"/>
  <vType id="truck" length="12.0" width="2.5" accel="1.0" decel="2.5" vClass="truck"/>
  <vType id="bike" length="1.8" width="0.65" accel="1.2" decel="3.0" maxSpeed="6.0" vClass="bicycle"/>
</vTypes>
<routes>
  <vehicle id="v1" type="car" depart="0" route="r1"/>
  <vehicle id="v2" type="truck" depart="5" route="r2"/>
  <vehicle id="v3" type="bike" depart="10" route="r3"/>
</routes>
```

**LibSignal SUMO integration**: 
- Converter (`common/converter.py`) reads `vType` from SUMO files (line 700-706)
- State features aggregate **all** vehicles (no type filtering currently)
- **To differentiate by type**: Modify `world_sumo.py` state queries to filter by `eng.vehicle.getTypeID(veh_id)`

#### **CityFlow: Partial Support ⚠️**
- Each flow defines vehicle params but **no explicit type system**
- All vehicles in a flow share same parameters
- **Workaround**: Create separate flows for each vehicle type:
  ```json
  [
    {"vehicle": {"length": 5.0, ...}, "route": [...], ...},    // Cars
    {"vehicle": {"length": 12.0, ...}, "route": [...], ...}   // Trucks
  ]
  ```
- **Limitation**: CityFlow doesn't distinguish vehicle classes in simulation (no bikes, no lane restrictions)
- **Physical diversity works** (length, accel), **behavioral diversity limited**

### How to Implement Heterogeneous Traffic

#### **Step 1: Define Vehicle Types in Traffic Files**

**SUMO** (`data/raw_data/{network}/{network}.rou.xml`):
```xml
<routes>
  <vType id="car" length="5.0" width="2.0" accel="2.6" decel="4.5" vClass="passenger"/>
  <vType id="truck" length="12.0" width="2.5" accel="1.5" decel="3.0" vClass="truck"/>
  <vType id="bike" length="1.8" width="0.65" accel="1.2" maxSpeed="6.0" vClass="bicycle"/>
  
  <flow id="flow_cars" type="car" begin="0" end="3600" probability="0.2" from="edge1" to="edge5"/>
  <flow id="flow_trucks" type="truck" begin="0" end="3600" probability="0.05" from="edge2" to="edge6"/>
  <flow id="flow_bikes" type="bike" begin="0" end="3600" probability="0.1" from="edge3" to="edge7"/>
</routes>
```

**CityFlow** (`data/raw_data/{network}/flow.json`):
```json
[
  {
    "vehicle": {"length": 5.0, "width": 2.0, "maxPosAcc": 2.6, ...},
    "route": [...],
    "interval": 5,
    "startTime": 0,
    "endTime": 3600
  },
  {
    "vehicle": {"length": 12.0, "width": 2.5, "maxPosAcc": 1.5, ...},
    "route": [...],
    "interval": 20,
    "startTime": 0,
    "endTime": 3600
  }
]
```

#### **Step 2: Modify State Extraction (Optional)**

To include vehicle type distribution in observations:

**Extend `world_sumo.py`** (add new state feature):
```python
def get_lane_vehicle_type_count(self):
    """Return dict: {lane_id: {'car': count, 'truck': count, 'bike': count}}"""
    result = {lane: {'car': 0, 'truck': 0, 'bike': 0} for lane in self.all_lanes}
    for lane in self.all_lanes:
        for veh in self.eng.lane.getLastStepVehicleIDs(lane):
            vtype = self.eng.vehicle.getTypeID(veh)
            if 'car' in vtype or 'passenger' in vtype:
                result[lane]['car'] += 1
            elif 'truck' in vtype:
                result[lane]['truck'] += 1
            elif 'bike' in vtype or 'bicycle' in vtype:
                result[lane]['bike'] += 1
    return result
```

**Register in `world_sumo.py` fns dict**:
```python
self.fns = {
    # ... existing fns
    "lane_vehicle_type_count": self.get_lane_vehicle_type_count,
}
```

**Use in agent** (`agent/dqn.py`):
```python
self.ob_generator = LaneVehicleGenerator(
    self.world, self.inter,
    ['lane_count', 'lane_vehicle_type_count'],  # Now includes type distribution
    in_only=True, average=None
)
```

#### **Step 3: Test**

Run with SUMO:
```bash
python run.py --agent dqn --world sumo --network {heterogeneous_network} --interface traci
```

Verify vehicle types in SUMO GUI (`--gui True` in config, but requires SUMO_HOME path fix).

---

## 4. Reaction Time Implementation

**Reaction time** = delay between light change and driver response.

### Current State

**No explicit reaction time modeling**:
- Vehicles respond instantly when phase changes
- Car-following models (SUMO IDM, CityFlow default) have implicit acceleration delays but not cognitive reaction time

### Implementation Strategies

#### **Option 1: SUMO Built-in Reaction Time**

SUMO supports **actionStepLength** (driver reaction time):

**In `vType` definition**:
```xml
<vType id="car" length="5.0" accel="2.6" decel="4.5" actionStepLength="1.0"/>
<!-- actionStepLength=1.0 → driver reacts every 1 second (vs default 0.1s) -->
```

**Per-vehicle**:
```xml
<vehicle id="v1" type="car" depart="0" route="r1">
  <param key="actionStepLength" value="1.5"/>  <!-- 1.5s reaction time -->
</vehicle>
```

**Heterogeneous reaction times**:
```xml
<vType id="young_driver" actionStepLength="0.8"/>
<vType id="old_driver" actionStepLength="1.5"/>
```

**How it works**: Vehicle updates acceleration/lane-change decisions every `actionStepLength` seconds instead of every simulation step.

**LibSignal integration**: Works automatically if defined in `.rou.xml` or `vType` files.

#### **Option 2: CityFlow Headway Time (Approximate)**

CityFlow has `headwayTime` in vehicle params:
```json
"vehicle": {
  "headwayTime": 2.0  // Time gap maintained behind lead vehicle (seconds)
}
```

This affects following distance but **not true reaction time** (vehicle still responds instantly to speed changes).

**Limitation**: CityFlow lacks dedicated reaction time parameter.

**Workaround**: Increase `headwayTime` to simulate cautious/delayed drivers (indirect effect).

#### **Option 3: Custom Implementation (World Wrapper)**

For precise control, wrap world's `step()` to delay phase changes:

**Modify `world_sumo.py`**:
```python
class Intersection:
    def __init__(self, ...):
        # ... existing code
        self.reaction_time_steps = 10  # 10 steps × 0.1s = 1s reaction delay
        self.pending_phase_change = None
        self.reaction_counter = 0
    
    def set_signal(self, action, yellow_time):
        """Queue phase change instead of applying immediately"""
        if action != self.current_phase:
            self.pending_phase_change = action
            self.reaction_counter = self.reaction_time_steps
        # Don't apply immediately; wait for reaction_counter to expire
    
    def step(self):
        """Apply pending phase change after reaction delay"""
        if self.reaction_counter > 0:
            self.reaction_counter -= 1
            if self.reaction_counter == 0 and self.pending_phase_change is not None:
                # Now apply the phase change
                self._apply_phase_change(self.pending_phase_change, yellow_time)
                self.pending_phase_change = None
```

**Trade-off**: More control but requires modifying core code.

#### **Option 4: Agent-Level Delayed Actions**

Simplest: Agent holds action for N steps before executing.

**Modify `trainer/tsc_trainer.py`**:
```python
# In train() loop
if i % self.action_interval == 0:
    actions = [ag.get_action(obs[idx], phase[idx]) for idx, ag in enumerate(self.agents)]
    
    # Store action but don't execute yet
    if not hasattr(self, 'action_queue'):
        self.action_queue = deque(maxlen=reaction_time_steps)
    self.action_queue.append(actions)
    
    # Execute delayed action
    if len(self.action_queue) == reaction_time_steps:
        delayed_actions = self.action_queue[0]
        obs, rewards, dones, _ = self.env.step(delayed_actions.flatten())
```

**Trade-off**: Simple but doesn't model per-vehicle heterogeneity.

### Recommendation

- **SUMO**: Use `actionStepLength` in `vType` definitions (easiest, most realistic)
- **CityFlow**: Limited options; use `headwayTime` or skip (CityFlow less suitable for reaction time studies)
- **Research-grade**: Custom wrapper (Option 3) for fine-grained control

---

## 5. Alternative Traffic Dynamics Approaches

Beyond state features and reaction time, how else can you model traffic dynamics?

### **A. Traffic Flow Models**

#### **Macroscopic (Flow-Based)**
- **Cell Transmission Model (CTM)**: Discretizes roads into cells, models density propagation
- **LWR Model**: Partial differential equation for traffic density waves
- **Not in LibSignal**: Current implementation uses microscopic simulation (individual vehicles)
- **Potential use**: Reward shaping based on CTM predictions (e.g., predict queue buildup)

#### **Mesoscopic (Speed-Based)**
- **SUMO Mesoscopic Mode**: Simulates vehicle platoons instead of individuals (faster, less detailed)
- **LibSignal support**: Partial (would need `world_sumo.py` modifications to use `eng.edge.getLastStepMeanSpeed()` instead of vehicle-level queries)

### **B. Car-Following & Lane-Change Models**

#### **SUMO Car-Following Models** (already supported via config):
- **IDM (Intelligent Driver Model)**: Realistic acceleration based on gap, speed difference
- **Krauss**: Original SUMO model (safe speed calculation)
- **W99**: Wiedemann 99 (German highway model)
- **Specify in vType**:
  ```xml
  <vType id="car" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.0"/>
  ```

#### **Lane-Change Models**:
- **LC2013**: Default SUMO lane change (cooperative)
- **SL2015**: Sublane model (lateral movement within lane)
- **Specify**:
  ```xml
  <vType id="aggressive" laneChangeModel="SL2015" lcStrategic="1.5" lcCooperative="0.5"/>
  ```

#### **CityFlow**: Fixed models (no user selection)

### **C. Dynamic Traffic Assignment (DTA)**

**Not in LibSignal**: Vehicles follow fixed routes

**Extension**: Integrate SUMO's `duarouter` for route choice based on RL agent's decisions:
1. Agent adjusts signal timings
2. SUMO re-routes vehicles to minimize travel time (dynamic user equilibrium)
3. Feedback loop: Better signals → better routes → different traffic patterns

**Implementation**: Run `duarouter` periodically in training loop (computationally expensive).

### **D. Incident Modeling**

Simulate accidents, road closures, events:

**SUMO**: Use `<vaporizer>` (vehicle removal) or `<closingReroute>` in additional files
```xml
<additional>
  <vaporizer id="accident_zone" begin="900" end="1200" edge="edge_5"/>
  <!-- Remove all vehicles on edge_5 between 15-20 min (accident) -->
</additional>
```

**LibSignal**: Load `.add.xml` in `configs/sim/{network}.cfg`:
```json
{
  "sumocfg": "network.sumocfg",
  "additional": "incidents.add.xml"
}
```

### **E. Pedestrian & Bike Interactions**

**SUMO**: Full pedestrian simulation with crossings
```xml
<vType id="pedestrian" vClass="pedestrian" width="0.5" length="0.5" maxSpeed="1.5"/>
<person id="p1" depart="0">
  <walk edges="edge1 edge2"/>
</person>
```

**Signals for pedestrians**: Add pedestrian phases to traffic light
```xml
<tlLogic id="junction1" type="static">
  <phase duration="30" state="GGGrrr"/>  <!-- Vehicles green -->
  <phase duration="5" state="yyyrrr"/>   <!-- Vehicles yellow -->
  <phase duration="20" state="rrrGGG"/>  <!-- Pedestrians green -->
</tlLogic>
```

**LibSignal state**: Modify generators to include pedestrian counts (similar to vehicle type extraction)

### **F. Weather & Visibility**

**SUMO**: Friction coefficient affects braking
```xml
<vType id="car_wet_road" accel="2.0" decel="3.0"/>  <!-- Reduced decel in rain -->
```

**LibSignal**: Switch vTypes mid-episode or create weather-specific networks

### **G. V2X Communication**

**Not natively supported**: Would require custom state features

**Example**: Include upstream traffic info in observation
- Modify `ob_generator` to query lanes beyond immediate intersection
- Agent sees "2 intersections upstream has queue of 50 vehicles"

**Implementation**:
```python
# In generator/lane_vehicle.py
def get_extended_observation(self):
    obs = []
    for intersection in [self.I] + self.I.upstream_neighbors:
        obs.extend(self.get_lane_counts(intersection))
    return obs
```

### **H. Adaptive Speed Limits**

**SUMO**: Dynamic speed limit changes
```xml
<variableSpeedSign id="vss1" lanes="edge1_0 edge1_1" pos="100">
  <step time="900" speed="13.89"/>   <!-- 50 km/h at t=900s -->
  <step time="1800" speed="8.33"/>   <!-- 30 km/h at t=1800s -->
</variableSpeedSign>
```

**LibSignal extension**: Agent action space includes speed limit adjustments (in addition to phase)

---

## Summary

### Key Takeaways

1. **Folder roles**: `agent/` (policies), `world/` (simulators), `generator/` (state extractors), `trainer/` (orchestration)
2. **State/reward fairness**: LibSignal allows per-agent definitions → unfair comparisons unless standardized
3. **Heterogeneous traffic**:
   - SUMO: Full support via `vType` system
   - CityFlow: Partial (physical params only, no behavioral classes)
   - Modify `world_sumo.py` to extract type-specific states
4. **Reaction time**:
   - SUMO: Use `actionStepLength` in vTypes (recommended)
   - CityFlow: Limited; approximate with `headwayTime`
   - Custom: Wrap world step for delays
5. **Traffic dynamics beyond LibSignal**:
   - Macroscopic models (CTM, LWR) for reward shaping
   - Car-following/lane-change models in SUMO (already configurable)
   - DTA, incidents, pedestrians, weather (SUMO supports; needs LibSignal integration)
   - V2X, adaptive speed limits (requires custom extensions)

### Recommended Next Steps

- **Standardize experiments**: Fix `ob_generator` and `reward_generator` params across agents for fair comparison
- **Exploit SUMO features**: Use `vType`, `actionStepLength`, pedestrian simulation for richer scenarios
- **Extend state extraction**: Add vehicle type counts, upstream traffic info to observations
- **Document assumptions**: Always report state/reward choices and simulator settings in results
