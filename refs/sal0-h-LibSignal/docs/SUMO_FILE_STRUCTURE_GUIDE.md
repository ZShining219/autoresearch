# SUMO File Structure Guide for Upper Bound Optimization

## Overview
For computing a theoretical upper bound (knowing all routes in advance, all network info, only controlling lights), you need to understand:

1. **Network files** (`roadnet.json`) - network topology, intersections, roads, phases
2. **Flow/Route files** (`flow.json`) - vehicle routes, departure times, vehicle properties
3. **How the code reads them** - where to hook in

---

## 1. Network File Structure (`roadnet.json`)

### Root Level
```json
{
  "intersections": [...],    // All traffic lights and virtual nodes
  "roads": [...]             // All edges in network
}
```

### Intersections (Traffic Lights)
Each intersection with a traffic light:
```json
{
  "id": "intersection_1_1",              // Unique ID
  "point": {"x": 0, "y": 0},            // 2D coordinates
  "width": 10,                           // Physical size
  "roads": [                             // Connected road IDs
    "road_0_1_0",
    "road_1_0_1",
    "road_2_1_2",
    ...
  ],
  "roadLinks": [                         // Turnmovement definitions
    {
      "type": "go_straight",             // Type: straight, turn_right, turn_left
      "startRoad": "road_0_1_0",         // From this incoming road
      "endRoad": "road_1_1_0",           // To this outgoing road
      "direction": 0,                    // Direction (0=N, 1=E, 2=S, 3=W)
      "laneLinks": [                     // Which lanes can turn
        {
          "startLaneIndex": 1,           // Which lane in startRoad
          "endLaneIndex": 0              // Which lane in endRoad
        }
      ]
    },
    // More turn movements...
  ],
  "trafficLight": {
    "lightphases": [                     // All possible signal phases
      {
        "time": 30,                      // Default phase duration (seconds)
        "availableRoadLinks": [0, 1, 2]  // Which roadLinks are green in this phase
      },
      // More phases...
    ]
  }
}
```

### Key Concepts for Optimization
- **Road**: Directional edge from node A to node B
- **Lane**: Within a road, vehicles are in lanes (e.g., left lane, right lane)
- **Turn movement**: Road1 → Road2 path through intersection
- **Phase**: A signal state that enables certain turn movements simultaneously
- **roadLinks[i]** maps to **lightphases[j].availableRoadLinks** to define what's green

### Roads
```json
{
  "roads": [
    {
      "id": "road_0_1_0",                // Unique ID (format often: road_from_to_direction)
      "startIntersection": "intersection_0_1",
      "endIntersection": "intersection_1_1",
      "length": 300,                     // Road length in meters
      "lanes": [                         // Number of lanes
        {
          "width": 3.5,
          "maxSpeed": 11.11              // Max speed (m/s)
        },
        // More lanes...
      ]
    },
    // More roads...
  ]
}
```

---

## 2. Flow/Route File Structure (`flow.json`)

### High-level Format
```json
[
  {
    "vehicle": {
      "length": 5.0,                     // Vehicle physical length (m)
      "width": 2.0,
      "maxPosAcc": 2.0,                  // Max acceleration (m/s²)
      "maxNegAcc": 4.5,                  // Max deceleration (m/s²)
      "minGap": 2.5,                     // Safe following distance (m)
      "maxSpeed": 11.11                  // Max speed (m/s)
    },
    "route": [                           // Deterministic route for all vehicles with this pattern
      "road_0_1_0",
      "road_1_1_0"
    ],
    "interval": 18.0,                    // Time between vehicle departures (seconds)
    "startTime": 0,                      // When to start generating (seconds)
    "endTime": 3600                      // When to stop generating (seconds)
  },
  // More demand patterns...
]
```

### What This Means
- **One entry = multiple vehicles** (interval 18s, startTime 0, endTime 3600 → ~200 vehicles)
- Departures: 0, 18, 36, 54, ... , 3600 seconds
- All follow the same route: road_0_1_0 → road_1_1_0
- You can compute **exact vehicle arrival times at each intersection**

### Computing Vehicle Arrivals
For each demand pattern:
1. Vehicle i departs at time: `depart_time = startTime + i * interval`
2. Vehicle follows route: `[road_0, road_1, road_2, ...]`
3. Travel time on each road = `road_length / max_speed` (approximately)
4. Arrival at intersection j = depart_time + sum(travel_times[0:j])

---

## 3. How LibSignal Reads These Files

### In Code
- [world/world_sumo.py](../world/world_sumo.py#L380): Reads JSON config
- [trainer/tsc_trainer.py](../trainer/tsc_trainer.py#L40): Sets simulation parameters

```python
# Example from world_sumo.py:
with open(sumo_config) as f:
    sumo_dict = json.load(f)

# sumo_dict contains:
# - dir: where roadnet.json and flow.json are located
# - roadnetFile: name of network file
# - flowFile: name of route file
```

### How to Access Network Data
```python
# During simulation (via TraCI interface):
self.eng.edge.getIDList()              # All road IDs
self.eng.lane.getIDList()              # All lane IDs
self.eng.trafficlight.getIDList()      # All intersection IDs
self.eng.trafficlight.getControlledLinks(intersection_id)  # Turn movements
self.eng.edge.getLength('road_0_1_0')  # Road length
self.eng.lane.getMaxSpeed('road_0_1_0_0')  # Lane max speed
```

---

## 4. For Upper Bound Computation

### What You Know in Advance
1. **Network Topology**
   - All intersections and their locations
   - All roads and lanes
   - All turn movements and which are compatible (can be green simultaneously)
   - Road lengths and max speeds

2. **Vehicle Routes**
   - Exact departure times for each vehicle (from flow.json)
   - Exact path each vehicle will take
   - Vehicle dynamics (acceleration, deceleration, max speed)

3. **Control Authority**
   - You can change signal phase at any intersection
   - Phases must satisfy: compatible turn movements are green together
   - You want to minimize travel time (or delay, or queue)

### Optimization Problem Formulation

**Inputs:**
- Network graph G = (intersections, roads)
- For each road: length, lanes, max speed → travel time
- For each vehicle v: departure_time, route [r₁, r₂, ..., rₙ]
- For each intersection i: available phases (disjoint sets of turn movements)

**Variables:**
- `phase[i, t]`: which phase is active at intersection i at time t

**Objective:**
- Minimize total travel time or delay: `Σ_v (arrival_time[v] - departure_time[v])`

**Constraints:**
- Phase continuity: valid transitions between phases
- Minimum phase duration: each phase lasts ≥ yellow_time + min_green
- Capacity: roads don't overflow (or teleport vehicles if capacity exceeded)

### Solution Approaches

**Exact (Small Networks):**
1. Mixed Integer Linear Programming (MILP)
2. Dynamic Programming (if temporal structure allows)

**Approximate (Larger Networks):**
1. Greedy: at each intersection, always choose phase that clears most waiting vehicles
2. Rollout: simulate 10-step lookahead, pick best phase
3. Genetic Algorithm: evolve phase schedules

---

## 5. Practical: Extract Network Data from sumo1x1

```python
import json

# Read network
with open('data/raw_data/syn_1x1_uniform_1600_1h/roadnet.json') as f:
    net = json.load(f)

# Extract key info
intersections = {inter['id']: inter for inter in net['intersections']}
roads = {road['id']: road for road in net['roads']}

# For each intersection, get phases
for int_id, inter in intersections.items():
    phases = inter['trafficLight']['lightphases']
    print(f"{int_id}: {len(phases)} phases")
    for i, phase in enumerate(phases):
        compatible_turns = phase['availableRoadLinks']
        print(f"  Phase {i}: roads {compatible_turns}")

# Read flow
with open('data/raw_data/syn_1x1_uniform_1600_1h/flow.json') as f:
    flows = json.load(f)

# Compute vehicles
total_vehicles = 0
for demand in flows:
    num_vehicles = int((demand['endTime'] - demand['startTime']) / demand['interval'])
    total_vehicles += num_vehicles
    route = demand['route']
    interval = demand['interval']
    print(f"Route {route}: {num_vehicles} vehicles, interval {interval}s")

print(f"Total vehicles: {total_vehicles}")
```

---

## 6. File Locations for All Networks

| Network | Path | Type |
|---------|------|------|
| sumo1x1 (synthetic) | `data/raw_data/syn_1x1_uniform_1600_1h/` | Uniform traffic |
| sumo4x4 (synthetic) | `data/raw_data/syn_4x4_gaussian_500_1h/` | Gaussian peaks |
| sumohz1x1 (real) | `data/raw_data/hangzhou_1x1_*/` | Real Hangzhou data |
| sumohz4x4 (real) | `data/raw_data/hangzhou_4x4_*/` | Real Hangzhou 4×4 |
| Cologne1x1 | `data/raw_data/cologne1/` | Real Cologne data |

All follow the same structure: `roadnet.json`, `flow.json`, optional `signal_plan_template.txt`

---

## 7. Key Takeaway for Your Upper Bound

Since you know:
- All vehicles' routes in advance ✓
- All network topology ✓
- Only can control signal phases ✓

The upper bound is: **Optimal offline schedule** where you tell each intersection *exactly which phase to be in at each time step*, minimizing total travel time.

Real RL agents must learn this without knowing future vehicle arrivals → they will never beat the offline upper bound, but that's expected.

See how much better RL gets compared to this theoretical maximum!
