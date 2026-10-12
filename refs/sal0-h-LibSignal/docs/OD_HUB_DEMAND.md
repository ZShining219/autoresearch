# Hub-centric OD demand realism (grid4x4, 1800 s)

## Claim

Score RL-TSC on **held-out hub-centric demand**, not one training movie. Compose with `realism_full` for Level 2.

## Generate demand set

```bash
python extras/gen_od_hub_demand_grid4x4.py
python extras/gen_od_hub_demand_grid4x4.py --validate-only
```

Outputs: `data/raw_data/grid4x4/od_hubs/demand_set/` (`fixed_1800`, `train_*`, `hold_*`, `manifest.json`).

TAZ: `data/raw_data/grid4x4/od_hubs/taz.xml` (fringe + internal + hub). Routing: `duarouter` shortest path. Departures: random via `od2trips`.

## LibSignal configs

| Level | Agents | Config suffix |
|-------|--------|----------------|
| 1 demand only | FT, MP, DQN, PressLight, CoLight | `*_odh_l1` |
| 2 demand + realism_full | same | `*_odh_l2` |

Shared protocol: [`configs/tsc/od_hub_1800_base.yml`](../configs/tsc/od_hub_1800_base.yml) (1800 s, demand set rotation, held-out every 10).

```bash
# Level 1 baselines
python run.py -a maxpressure_odh_l1 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l1
python run.py -a fixedtime_odh_l1 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l1

# Level 1 RL (100 episodes)
python run.py -a dqn_odh_l1 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l1
python run.py -a presslight_odh_l1 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l1
python run.py -a colight_odh_l1 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l1

# Level 2 (same + realism_full)
python run.py -a maxpressure_odh_l2 -w sumo -n sumo4x4 --seed 42 --ngpu -1 --prefix odh_l2
# ... fixedtime_odh_l2, dqn_odh_l2, presslight_odh_l2, colight_odh_l2
```

Or: `./extras/submit_od_hub_1800.sh l1` / `l2` / `all`.

Primary metric: `HELDOUT_MEAN` in BRF/DTL logs.
