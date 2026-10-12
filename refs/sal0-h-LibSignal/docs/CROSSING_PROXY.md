# Crossing Proxy — Actuated Pedestrian-Crossing Delay (Axis 5)

Fifth realism axis for the realism-decomposed RL-TSC benchmark. Models **NEMA
concurrent pedestrian service** at signalized intersections without explicit SUMO
`<person>` entities.

See also: `docs/PARTIAL_OBSERVABILITY.md` (axis 4), `docs/SIGNAL_CONTROL_THEORY.md` §9.

---

## Real-world mechanism (what we implement)

In NEMA timing, a **concurrent pedestrian walk** runs with the **parallel through
phase** (e.g. N–S through green with E–W crosswalk walk). That creates two vehicle
effects:

1. **Actuated phase extension (primary)** — a ped call **holds the current through
   green** for walk + flashing-don’t-walk (FDW) + clearance (calibrated as
   `crossing_service_min` … `crossing_service_max` seconds). The controller cannot
   transition out until service ends. This lengthens the cycle and increases wait
   for all other movements — the main network-wide ATT impact.

2. **Crosswalk–vehicle conflicts (secondary, when present)** — any **non-red**
   movement on the **crosswalk street** (G / g / s) yields to pedestrians. On
   protected grids (grid4x4 post PR #6) cross-street through is red during concurrent
   walk, so this list is often **empty**; permissive/right-turn networks populate
   `conflict_lanes` in the JSON map and those lanes are halted for the same service
   interval.

We do **not** halt cross-street approaches that are already red (no real-world effect).

---

## What this axis does not model

- Pedestrian agents, sidewalks, spatial crossing, Barnes dance, ped counts in obs
- Separate ped phases or longer TLS state strings
- Pedestrian delay metrics (primary metric remains **vehicle ATT**)

**Paper label:** *actuated concurrent-walk proxy* — not full multimodal simulation.

Full SUMO pedestrians remain a future extension (`docs/TECHNICAL_ANALYSIS.md` §E).

---

## Runtime behaviour (`world/world_sumo.py`)

1. Load `crossing_proxy_lanes.json` (through_incoming + conflict_lanes per phase 0/4).
2. **Before each `pseudo_step`** (same second as the decision):
   - On entry to through phase 0 or 4, with probability `p` start service for
     `T ~ Uniform(7, 10)` seconds.
   - **Phase extension:** hold current green (`pseudo_step` cannot switch away).
   - **Physical yield:** `setMaxSpeed(lane, 0)` on all `through_incoming` lanes
     (driver/corner delay during concurrent walk) plus any `conflict_lanes`
     with G/g/s (turns). On protected grid4x4, conflict is often empty.
3. Logs `[CrossingProxy] ped calls=N ...` during run and total at episode end.

**Why earlier runs showed ~173 s:** (1) ped logic ran *after* `pseudo_step`, so
locks never blocked decisions; (2) phase-only extension has no effect when
MaxPressure already keeps the same green; (3) halting cross-street lanes that
are already red does nothing.

---

## Default parameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `crossing_call_prob` | **0.12** | Urban actuated ped calls on ~10–15% of through-green entries |
| `crossing_service_min` | **7.0 s** | Walk (~4 s on ~12 m @ 4 ft/s) + minimum FDW/clearance |
| `crossing_service_max` | **10.0 s** | Walk + typical FDW / clearance window (MUTCD-scale) |
| `crossing_seed_offset` | **7** | Separate RNG stream from demand / partial-obs |

---

## Configuration & run

```yaml
# configs/tsc/base.yml
world:
  crossing_proxy: false
  crossing_call_prob: 0.12
  crossing_service_min: 7.0
  crossing_service_max: 10.0
  crossing_seed_offset: 7
```

Regenerate lane map after net / signal-plan changes:

```bash
export SUMO_HOME="${SUMO_HOME:-/opt/homebrew/opt/sumo/share/sumo}"
python extras/gen_crossing_proxy_lanes.py --network grid4x4
```

```bash
python run.py -a maxpressure_crossing_proxy -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix crossing_proxy_smoke
```

Expect log: `[CrossingProxy] actuated extension — p=0.12, service=[7.0,10.0]s, ...`

Compare to homo MP **173.26 s** / FT **219.14 s** (corrected 4×4).

---

## Composability

Composable with `hetero`, `slow_start`, and partial observability (no new exclusivity rule).

---

## Files

| File | Role |
|------|------|
| `world/world_sumo.py` | `CrossingProxyController`, phase lock in `pseudo_step` |
| `extras/gen_crossing_proxy_lanes.py` | Conflict-lane map from net + phase states |
| `data/raw_data/grid4x4/crossing_proxy_lanes.json` | Generated artifact |
| `configs/tsc/*_crossing_proxy.yml` | Enable flag per agent |
