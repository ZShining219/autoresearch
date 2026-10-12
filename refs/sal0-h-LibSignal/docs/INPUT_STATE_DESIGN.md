# Rich input variants

## Purpose and scope

Compare each learned controller with a version that receives more local traffic
information. The registered agents are `dqn_rich`, `presslight_rich`, and
`colight_rich`. These are separate experimental variants of the same learning
algorithms and must be trained from scratch. There is no global observation flag:
support for another algorithm requires an explicit implementation and config.

FixedTime, MaxPressure, and Traffic-R1 are unchanged. These variants support SUMO.

## Observation definition

`generator/rich_lane_vehicle.py` builds four contiguous blocks:

```text
[incoming counts | incoming stopped counts | incoming mean speeds | outgoing counts]
```

For an intersection with `I` incoming lanes and `O` outgoing lanes, this is
`3*I + O` values, before any phase feature is added by the parent agent.

| Block | Definition | Units |
| --- | --- | --- |
| Incoming counts | Existing `world.get_info('lane_count')` values on incoming lanes | Vehicles |
| Incoming stopped counts | Visible vehicles whose current speed is strictly below 0.1 m/s | Vehicles |
| Incoming mean speeds | Arithmetic mean of visible vehicles' instantaneous speeds; 0 when none are visible | m/s |
| Outgoing counts | Existing `lane_count` values on outgoing lanes | Vehicles |

Road order follows the intersection's existing `in_roads` and `out_roads` order.
Within a road, lanes follow numeric SUMO lane index order, reversed when
`world.RIGHT` is false. Counts are not normalized. Speeds remain in m/s; CoLight's
existing `vehicle_max` divisor is inherited and equals 1 in the supplied configs.
The meaning and scale of these features are the same across all three variants.

Stopped counts deliberately do not use the current `queue_length` or
`lane_waiting_count` fields. In this fork, `queue_length` counts all observed
vehicles, while `lane_waiting_count` includes vehicles with recorded waiting time.
A vehicle that has resumed moving is excluded from the new stopped count.

## Visibility and noise

The generator reads the existing per-intersection `full_observation` snapshot; it
does not query extra vehicles from SUMO. The current world path filters vehicles
by the next traffic light being within 200 m and by persistent per-vehicle
`obs_penetration`. That same coverage applies to speed and stopped counts.
Outgoing counts inherit this existing detector coverage, including its next-light
filter; they are not whole-link occupancy or downstream free-space measurements.

Incoming and outgoing total counts reuse the world's already corrupted counts.
Stopped counts use the same additive/proportional Gaussian noise setting and
rounding/nonnegative clamp, through `World._noisy_count`. Their random stream is
keyed by observation seed, integer simulation time, and lane ID, separately from
the existing count, reward, and simulation streams. Repeated reads at the same
step are identical. No noise is added when the configured standard deviation is
zero. Independently noisy total and stopped counts need not obey stopped <= total;
additive noise can also report a positive count on an empty visible lane.

Mean speeds are computed after visibility filtering, without an additional speed
noise model. Thus L2 corrupts the count channels and limits the vehicles used in
all channels; it does not represent a general model of speed sensor error.

## Agent integration

Each parent has one `_make_observation_generator` hook, used in construction and
reset. Its default implementation returns exactly the previous generator.
Each rich subclass replaces that hook with `RichLaneVehicleGenerator`.

| Agent | Native policy input | Rich policy input | Retained behavior |
| --- | --- | --- | --- |
| DQN | Incoming counts + current phase one-hot | Four blocks + current phase one-hot | Per-intersection DQN, waiting-count reward, phase-selection actions |
| PressLight | Incoming and outgoing counts + current phase one-hot | Four blocks + current phase one-hot | Per-intersection DQN, pressure reward, phase-selection actions |
| CoLight | Incoming counts at each graph node | Four blocks at each graph node | Same graph/attention, waiting-count reward, masked phase-selection actions, no phase input |

DQN and PressLight retain their intersection-specific input sizes. CoLight needs
one width across graph nodes: `colight_rich` computes the maximum incoming and
outgoing lane counts, then pads each block separately with zeros. This keeps
stopped/speed/outgoing offsets consistent on intersections of different sizes.
Graph connections, node ordering, attention layers, and action masks are inherited.

At a regular 4x4 interior intersection, `I=O=12` and there are eight phases:
DQN and PressLight receive 56 values; CoLight receives 48 values per node. The
grid also has 16 controlled boundary signals in addition to its 16 interior
signals; the existing trainer still creates controllers for all 32.

Rewards, action intervals, phase transition logic, replay buffers, learning
updates, hidden layers, and optimizers use the parent implementations. Only the
input layer widens to accept the new features, so parameter counts also increase.
This experiment isolates an input-design intervention within each algorithm, not
an equal-parameter-count comparison. Native and rich checkpoints are incompatible.

## Configs and use

Each rich config includes the corresponding native config and overrides only
`model.name`. This preserves each level's demand, realism settings, and training
protocol. The `--agent` argument names the config as it does for existing runs.

| Protocol | Example config name (`--agent`) |
| --- | --- |
| Native base / L0 settings | `dqn_rich` |
| Grid L1 | `dqn_rich_odh_l1` |
| Grid L2 | `dqn_rich_odh_l2` |
| Ingolstadt L1 | `dqn_rich_odh_l1_1x21` |
| Ingolstadt L2 | `dqn_rich_odh_l2_1x21` |

Replace `dqn` with `presslight` or `colight` for the other variants. Use
`--network sumo4x4` for the grid and `--network sumo1x21` for Ingolstadt. Outputs
and checkpoints use the existing directory scheme with the rich config name,
keeping native and rich runs separate. No existing checkpoint is migrated.

For the paper, compare native versus rich within each algorithm under the same
level, demand files, seeds, metrics, and training protocol. CoLight still has
graph communication, and PressLight already had outgoing counts, so the added
information relative to each native baseline differs. List them as explicit
variants, and do not attribute every cross-algorithm difference to information.

## Bounded verification

Run from the repository root after activating the environment:

```bash
source .venv/bin/activate
python -W ignore::ResourceWarning -m unittest discover -s tests -v
python -m compileall -q agent world trainer task common utils generator dataset run.py environment.py
for network in sumo4x4 sumo1x21; do
  for agent in dqn_rich presslight_rich colight_rich; do
    python extras/smoke_rich_agents.py --agent "$agent" --network "$network" --ngpu -1
  done
done
```

The smoke script uses L2 with seed 42, runs only 300 simulated seconds, and then
performs three gradient updates per agent using a test-only batch size of 8.
It checks observation shapes, finite observations/rewards/losses, valid actions,
changed weights, exact checkpoint reload, and generator rebinding after reset.
It writes `smoke_result.json` under the run's `rich_smoke` output directory.
It does not call the trainer's episode loop or produce paper performance results.
Production configs retain their normal batch sizes and episode budgets.

Validated on 2026-09-12 using Python 3.12, PyTorch 2.5.1 CPU, and SUMO 1.26.0:
all 21 unit tests and compilation passed. All six L2 smoke runs passed the
rollout, gradient update, checkpoint, and reset checks above, with actual visible
traffic in every run. No full training run was launched.

| Variant | 4x4 policy input sizes | Ingolstadt policy input sizes | Smoke checks |
| --- | --- | --- | --- |
| DQN-rich | 5 (boundary), 56 (interior) | 18–53, depending on intersection | Passed on both |
| PressLight-rich | 5 (boundary), 56 (interior) | 18–53, depending on intersection | Passed on both |
| CoLight-rich | 48 per node | 50 per node | Passed on both; graph remapped to world order |

### CoLight graph mapping

On both tested networks, the native CoLight graph builder (`sumolib`
traffic-light order) and the SUMO world (`libsumo` `getIDList`) enumerate
signals differently. `agent/colight.py` remaps `sparse_adj` into
`world.intersections` rows so GNN neighbors, observations, rewards, phase
masks, and `env.step` share one index space. Generators stay in world order;
do not sort them into graph order. `colight_rich` inherits this remap.

A smoke run records `graph_order_matches_world` after checking that generator
ids equal `world.intersections` and that remapped edges index that range.
