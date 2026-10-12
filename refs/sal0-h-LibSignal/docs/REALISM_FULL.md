# Full realism capstone — all axes on (grid4x4)

Fifth realism axis plus composition of axes 1–4 into one deployment profile for
sim-to-real stress testing. Single-axis docs:

- Hetero fleet: `data/raw_data/vTypes_mixed.add.xml` / hetero routes
- Slow-start: `docs/` notes on feat/slow-start-v2
- Partial obs: `docs/PARTIAL_OBSERVABILITY.md`
- Crossing proxy: `docs/CROSSING_PROXY.md`

---

## Profile: `realism_full` (`configs/tsc/realism_full_world.yml`)

| Layer | Flag / param | Value | Effect |
|-------|----------------|-------|--------|
| **Plant — fleet** | `hetero: true` | 80% car / 20% truck | `grid4x4_hetero.rou.xml` |
| **Plant — discharge** | `slow_start: true` | combined vTypes | `vTypes_realism_full.add.xml` |
| **Plant — crossings** | `crossing_proxy: true` | p=0.12, 7–10 s | Phase hold + through-lane halts |
| **Sensor — penetration** | `obs_penetration` | **0.8** | 80% vehicles visible per trip |
| **Sensor — noise** | `obs_count_noise_std` | **2.0** additive | Noisy lane counts each step |

**Combined hetero + slow_start:** one route file (hetero mix) + one vType file where
**cars** get passenger slow-start (`accel=1.0`, `tau=1.9`) and **trucks** keep heavy-vehicle
geometry with **`accel=1.2`, `tau=2.1`** (longer headway, no passenger `pkw` override).

Agents never see vType; they read the same corrupted lane statistics as in single-axis runs.

---

## Fair comparison protocol

1. **Train and test RL under the same profile** (`*_realism_full` agents, 200 episodes).
2. **Baselines:** one 3600-step eval (`maxpressure_realism_full`, `fixedtime_realism_full`).
3. **Metrics:** ground-truth ATT (physics + crossing); observed queue/reward under partial obs.
4. **Sanity:** FixedTime ATT changes with physics axes; partial obs shifts observed queue only.
5. **Network:** `sumo4x4` only (crossing lane map exists). Seed **42**.

---

## Run commands

```bash
conda activate libsignal
export SUMO_HOME="$CONDA_PREFIX/share/sumo"
export PATH="$CONDA_PREFIX/bin:$SUMO_HOME/bin:$PATH"
cd ~/LibSignalFork

# Smoke — MaxPressure baseline (~12 min)
python run.py -a maxpressure_realism_full -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix realism_full_4x4

# RL (200 ep, ~12 h each on gpujobs)
python run.py -a dqn_realism_full -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix realism_full_4x4

python run.py -a presslight_realism_full -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix realism_full_4x4

python run.py -a colight_realism_full -w sumo -n sumo4x4 \
  --seed 42 --ngpu -1 --interface libsumo --prefix realism_full_4x4
```

**SLURM batch:**

```bash
export MCS_LABEL=crs-XXXX
./extras/submit_realism_full_4x4.sh all    # baselines + RL
./extras/submit_realism_full_4x4.sh rl     # DQN + PressLight + CoLight only
```

**Outputs:** `data/output_data/tsc/sumo_<agent>_realism_full/sumo4x4/realism_full_4x4/logger/`

---

## Expected interpretation

- ATT **above** any single-axis row; interactions may be **superlinear**.
- RL **train-on-profile**; do not evaluate homo-trained checkpoints here.
- Ranking may match single-axis (RL ≈ MP); capstone tests **compound stress**, not SOTA.
