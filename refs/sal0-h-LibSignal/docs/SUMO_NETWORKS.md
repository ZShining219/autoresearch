# SUMO Networks Available in LibSignal

All available SUMO networks with their paths, road network topology, and traffic flow files.

## Quick Reference

| Network Name | Type | Intersections | Config File | Road Network | Traffic Flow |
|---|---|---|---|---|---|
| sumo1x1 | Grid | 1×1 | [configs/sim/sumo1x1.cfg](../configs/sim/sumo1x1.cfg) | data/raw_data/cologne1/cologne1.net.xml | data/raw_data/cologne1/cologne1.rou.xml |
| sumo1x3 | Grid | 1×3 | [configs/sim/sumo1x3.cfg](../configs/sim/sumo1x3.cfg) | data/raw_data/cologne3/cologne3.net.xml | data/raw_data/cologne3/cologne3.rou.xml |
| sumo1x21 | Arterial | 1×21 | [configs/sim/sumo1x21.cfg](../configs/sim/sumo1x21.cfg) | data/raw_data/ingolstadt21/ingolstadt21.net.xml | data/raw_data/ingolstadt21/ingolstadt21.rou.xml |
| sumo4x4 | Grid | 4×4 interior (32 TLs) | [configs/sim/sumo4x4.cfg](../configs/sim/sumo4x4.cfg) | data/raw_data/grid4x4/grid4x4.net.xml | data/raw_data/grid4x4/grid4x4.rou.xml |
| sumo7x28 | Grid | 7×28 | [configs/sim/sumo7x28.cfg](../configs/sim/sumo7x28.cfg) | data/raw_data/manhattan_28x7/manhattan_28x7.net.xml | data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml |
| sumo1x1_colight | Grid | 1×1 (CoLight) | [configs/sim/sumo1x1_colight.cfg](../configs/sim/sumo1x1_colight.cfg) | data/raw_data/cologne1/cologne1.net.xml | data/raw_data/cologne1/cologne1.rou.xml |
| sumohz1x1 | Urban | 1×1 (Hangzhou) | [configs/sim/sumohz1x1.cfg](../configs/sim/sumohz1x1.cfg) | data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.net.xml | data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.rou.xml |
| sumohz1x1_config2 | Urban | 1×1 (Hangzhou v2) | [configs/sim/sumohz1x1_config2.cfg](../configs/sim/sumohz1x1_config2.cfg) | data/raw_data/hangzhou_1x1_qc-yn_18041608_1h/hangzhou_1x1_qc-yn_18041608_1h.net.xml | data/raw_data/hangzhou_1x1_qc-yn_18041608_1h/hangzhou_1x1_qc-yn_18041608_1h.rou.xml |
| sumohz1x1_config3 | Urban | 1×1 (Hangzhou v3) | [configs/sim/sumohz1x1_config3.cfg](../configs/sim/sumohz1x1_config3.cfg) | data/raw_data/hangzhou_1x1_kn-hz_18041608_1h/hangzhou_1x1_kn-hz_18041608_1h.net.xml | data/raw_data/hangzhou_1x1_kn-hz_18041608_1h/hangzhou_1x1_kn-hz_18041608_1h.rou.xml |
| sumohz1x1_config4 | Urban | 1×1 (Hangzhou v4) | [configs/sim/sumohz1x1_config4.cfg](../configs/sim/sumohz1x1_config4.cfg) | data/raw_data/hangzhou_1x1_sb-sx_18041607_1h/hangzhou_1x1_sb-sx_18041607_1h.net.xml | data/raw_data/hangzhou_1x1_sb-sx_18041607_1h/hangzhou_1x1_sb-sx_18041607_1h.rou.xml |
| sumohz4x4 | Urban | 4×4 (Hangzhou) | [configs/sim/sumohz4x4.cfg](../configs/sim/sumohz4x4.cfg) | data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml | data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.rou.xml |
| sumohz4x4_hetero | Urban | 4×4 (Hangzhou, heterogeneous) | [configs/sim/sumohz4x4_hetero.cfg](../configs/sim/sumohz4x4_hetero.cfg) | data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.net.xml | data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.rou.xml |

---

## Network Details

### Synthetic Grid Networks

#### **sumo1x1**
- **Topology**: Single 1×1 intersection (Cologne dataset)
- **Use case**: Minimal, quick tests
- **Path**:
  - Config: `configs/sim/sumo1x1.cfg`
  - Network: `data/raw_data/cologne1/cologne1.net.xml`
  - Flow: `data/raw_data/cologne1/cologne1.rou.xml`
- **Run**:
  ```bash
  python run.py --agent maxpressure --world sumo --network sumo1x1 --interface traci
  ```

#### **sumo1x3**
- **Topology**: Arterial corridor 1×3 intersections (Cologne `cologne3`; same map as `sumo_cologne3`)
- **Use case**: Multi-intersection learning, corridor flow patterns
- **Path**:
  - Config: `configs/sim/sumo1x3.cfg`
  - Network: `data/raw_data/cologne3/cologne3.net.xml`
  - Flow: `data/raw_data/cologne3/cologne3.rou.xml`

#### **sumo1x21**
- **Topology**: Long arterial 1×21 intersections (Ingolstadt `ingolstadt21`, 21 traffic lights)
- **Use case**: Arterial signal coordination, platoon formation
- **Path**:
  - Config: `configs/sim/sumo1x21.cfg`
  - Network: `data/raw_data/ingolstadt21/ingolstadt21.net.xml`
  - Flow: `data/raw_data/ingolstadt21/ingolstadt21.rou.xml`

#### **sumo4x4**
- **Topology**: Urban grid 4×4 interior (`A0`–`D3`) plus 16 fringe portal TLs
  (`top`/`bottom`/`left`/`right` `0–3`) — **32 traffic lights total**, so LibSignal
  creates **32 agents** (one per SUMO TL ID). See [AGENT_OBSERVATIONS.md](AGENT_OBSERVATIONS.md).
- **Use case**: Medium-scale multi-agent coordination
- **Path**:
  - Config: `configs/sim/sumo4x4.cfg`
  - Network: `data/raw_data/grid4x4/grid4x4.net.xml`
  - Flow: `data/raw_data/grid4x4/grid4x4.rou.xml`
- **Run**:
  ```bash
  python run.py --agent dqn --world sumo --network sumo4x4 --interface traci --seed 42
  ```

#### **sumo7x28**
- **Topology**: Large grid 7×28 (196 intersections; data folder `manhattan_28x7`)
- **Use case**: Large-scale experiments, scalability testing
- **Path**:
  - Config: `configs/sim/sumo7x28.cfg`
  - Network: `data/raw_data/manhattan_28x7/manhattan_28x7.net.xml`
  - Flow: `data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml`
- **Note**: Computationally intensive; requires CPU with many threads

#### **sumo1x1_colight**
- **Topology**: Single 1×1 intersection (Cologne, CoLight-optimized)
- **Use case**: Testing CoLight agent specifically
- **Path**:
  - Config: `configs/sim/sumo1x1_colight.cfg`
  - Network: `data/raw_data/cologne1/cologne1.net.xml`
  - Flow: `data/raw_data/cologne1/cologne1.rou.xml`

---

### Real-World Urban Networks (Hangzhou, China)

#### **sumohz1x1**
- **Topology**: Single 1×1 intersection (Hangzhou real traffic, default route)
- **Use case**: Real-world baseline, urban signal timing
- **Path**:
  - Config: `configs/sim/sumohz1x1.cfg`
  - Network: `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.net.xml`
  - Flow: `data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.rou.xml`
- **Data**: Real traffic from Hangzhou, collected 18:04–19:04 (peak hour)

#### **sumohz1x1_config2, config3, config4**
- **Topology**: Single 1×1 intersection (different Hangzhou routes/times)
- **Use case**: Cross-validation, time-of-day experiments
- **Routes**:
  - **config2**: `hangzhou_1x1_qc-yn_18041608_1h` (18:04–19:04, qc-yn site)
  - **config3**: `hangzhou_1x1_kn-hz_18041608_1h` (18:04–19:04, kn-hz site)
  - **config4**: `hangzhou_1x1_sb-sx_18041607_1h` (17:04–18:04, sb-sx site)

#### **sumohz4x4**
- **Topology**: Urban grid 4×4 (Hangzhou real traffic)
- **Use case**: Real-world multi-intersection coordination
- **Path**:
  - Config: `configs/sim/sumohz4x4.cfg`
  - Network: `data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml`
  - Flow: `data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.rou.xml`
- **Data**: Real Hangzhou traffic (18:04–19:04)

#### **sumohz4x4_hetero**
- **Topology**: Urban grid 4×4 (Hangzhou, heterogeneous traffic)
- **Use case**: Mixed vehicle types (cars, trucks, etc.)
- **Path**:
  - Config: `configs/sim/sumohz4x4_hetero.cfg`
  - Network: `data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.net.xml`
  - Flow: `data/raw_data/hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.rou.xml` (with vType variations)

---

## How to Run Experiments

### Quick Start (All Baselines on sumo1x1)
```bash
export SUMO_HOME=/home/salman/.conda/envs/traffic/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
conda activate traffic

# MaxPressure baseline
python run.py --agent maxpressure --world sumo --network sumo1x1 --interface traci

# SOTL baseline
python run.py --agent sotl --world sumo --network sumo1x1 --interface traci

# Fixed-time baseline
python run.py --agent fixedtime --world sumo --network sumo1x1 --interface traci
```

### Train DQN on Different Networks
```bash
# Small (1×1)
python run.py --agent dqn --world sumo --network sumo1x1 --seed 42 --ngpu 0 --interface traci

# Medium (4×4)
python run.py --agent dqn --world sumo --network sumo4x4 --seed 42 --ngpu 0 --interface traci

# Real-world (Hangzhou)
python run.py --agent dqn --world sumo --network sumohz1x1 --seed 42 --ngpu 0 --interface traci

# Large-scale (7×28)
python run.py --agent dqn --world sumo --network sumo7x28 --seed 42 --ngpu 0 --interface traci
```

### Specify Output Directory
```bash
python run.py --agent dqn --world sumo --network sumo4x4 --prefix exp_batch1 --seed 42 --interface traci
# Output: data/output_data/tsc/sumo_dqn_exp_batch1/
```

### Run Multiple Seeds
```bash
for seed in 1 2 3; do
  python run.py --agent dqn --world sumo --network sumo4x4 --seed $seed --interface traci
done
```

---

## Configuration File Structure

Each network's `.cfg` file specifies:
- **network**: Name/ID of topology
- **roadnetFile**: Path to SUMO `.net.xml` (road topology)
- **flowFile**: Path to SUMO `.rou.xml` (traffic routes)
- **interval**: Simulation step size (typically 1.0 second)
- **yellow_length**: Duration of yellow phase (seconds)
- **gui**: Whether to show SUMO GUI (`true`/`false`; requires X11 display)
- **combined_file**: SUMO config file (`.sumocfg`) if multi-file scenario

Example (`configs/sim/sumo4x4.cfg`):
```json
{
  "network": "grid4x4",
  "interval": 1.0,
  "seed": 0,
  "dir": "data/",
  "roadnetFile": "raw_data/grid4x4/grid4x4.net.xml",
  "flowFile": "raw_data/grid4x4/grid4x4.rou.xml",
  "no_warning": true,
  "name": "debug",
  "yellow_length": 3,
  "gui": false
}
```

---

## Network Selection Guide

| Goal | Recommended Network | Why |
|---|---|---|
| Quick test | `sumo1x1` | Instant runs, single intersection |
| Algorithm dev | `sumo1x1`, `sumo4x4` | Fast iteration, manageable |
| Baseline comp | `sumo4x4`, `sumohz1x1` | Standard benchmarks |
| Real-world eval | `sumohz4x4` | Actual traffic dynamics |
| Scalability study | `sumo7x28` | Large-scale complexity |
| Arterial signals | `sumo1x21` | Corridor coordination |
| Multi-vehicle types | `sumohz4x4_hetero` | Heterogeneous traffic |

---

## Common Issues

**"No SUMO in environment path"**
→ Set `SUMO_HOME` before running (see Quick Start above)

**Simulation very slow**
→ Running on CPU; add `--ngpu 0` to use available GPU, or reduce `--thread_num`

**Network not found error**
→ Verify `.net.xml` and `.rou.xml` exist in the paths above; check `data/raw_data/` directory

**TraCI connection refused**
→ SUMO process crashed; check for port conflicts or SUMO_HOME issues; retry with `--interface traci`
