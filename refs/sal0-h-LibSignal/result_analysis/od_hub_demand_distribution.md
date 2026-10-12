# Hub OD demand distribution checks (grid4x4, 1800 s)

This note summarizes whether the generated demand set under
`data/raw_data/grid4x4/od_hubs/demand_set/` matches the intended demand-realism
design (hub-centric OD, fringe/within mix, shoulder+peak timeline, stochastic
files). Analysis covers all 14 route files (`fixed_1800`, `train_00–09`,
`hold_00–02`).

Primary metric for controllers remains **held-out ATT** (`HELDOUT_MEAN`); this
document is about the **demand process**, not agent scores.

---

## Design vs realized composition

| Component | Target | Realized (14 files) |
|-----------|--------|---------------------|
| Fringe origins | ~60–70% | **65.1%** (range ~62–69%) |
| Within-network origins | ~30–40% | **34.9%** |
| Hub destinations | ~70% preference knob | **~88%** end at hub edges |
| Hub-touch (route hits a hub edge) | majority of trips | **~95%** |
| Volume | ~800 vehicles / file | **mean 794**, std ~26 |
| Episode length | 1800 s | 1800 s |

Hub-centrism is **stronger** than the 70% destination knob alone: gravity also
pulls flow toward hubs, and many non-hub OD pairs still **route through** hub
edges on the 4×4 grid.

### Per-file snapshot

| file | n | fringeO | intO | hubDest | hubTouch |
|------|--:|--------:|-----:|--------:|---------:|
| fixed_1800 | 796 | 0.646 | 0.354 | 0.868 | 0.941 |
| train_00 | 742 | 0.690 | 0.310 | 0.892 | 0.964 |
| train_01 | 779 | 0.638 | 0.362 | 0.872 | 0.949 |
| train_02 | 795 | 0.640 | 0.360 | 0.891 | 0.961 |
| train_03 | 849 | 0.668 | 0.332 | 0.890 | 0.969 |
| train_04 | 756 | 0.644 | 0.356 | 0.882 | 0.963 |
| train_05 | 790 | 0.638 | 0.362 | 0.871 | 0.951 |
| train_06 | 790 | 0.622 | 0.378 | 0.877 | 0.957 |
| train_07 | 810 | 0.644 | 0.356 | 0.872 | 0.944 |
| train_08 | 824 | 0.659 | 0.341 | 0.888 | 0.966 |
| train_09 | 792 | 0.616 | 0.384 | 0.872 | 0.960 |
| hold_00 | 781 | 0.662 | 0.338 | 0.883 | 0.951 |
| hold_01 | 818 | 0.671 | 0.329 | 0.890 | 0.956 |
| hold_02 | 796 | 0.670 | 0.330 | 0.866 | 0.946 |

`hubO` (trips that *originate* on hub edges) is ~0 by construction: hubs are
attractors; origins are fringe or internal sources.

---

## Timeline (shoulder + peak)

Trips are allocated into six 300 s bins with fixed weights, then `od2trips`
places departures inside each bin.

| Bin (s) | Design share | Realized share (pooled) | ratio |
|---------|-------------:|------------------------:|------:|
| 0–300 | 0.08 | 0.080 | ~1.00 |
| 300–600 | 0.12 | 0.120 | ~1.00 |
| 600–900 | 0.22 | 0.220 | ~1.00 |
| 900–1200 (peak) | 0.28 | 0.280 | ~1.00 |
| 1200–1500 | 0.18 | 0.180 | ~1.00 |
| 1500–1800 | 0.12 | 0.120 | ~1.00 |

The departure **volume profile** matches the intended non-flat day.

---

## Randomness / “Poisson”

### What the generator actually does

1. Sample OD counts with RNG (fringe/internal mix, hub destination bias, gravity).
2. Split trips across timeline bins by weight.
3. Call SUMO `od2trips` **without** `--spread.uniform` → random departures inside each bin.
4. Call `duarouter` for shortest-path routes on `grid4x4.net.xml`.
5. Draw total vehicles per file ≈ `Normal(800, 25)` (seeded).

### What it does *not* do

- Does **not** pass `od2trips --flow-output.poisson`.
- Total demand per file is **Gaussian around 800**, not a Poisson count of trips.
- So: describe arrivals as **random-in-bin via od2trips**, not a calibrated Poisson process.

### Within-bin inter-departure gaps (`fixed_1800`)

For a homogeneous Poisson process, gaps are exponential: CV ≈ 1 and
P(gap &lt; mean) ≈ 0.632.

| Bin | n departures | mean gap | CV (Exp ≈ 1) | P(gap &lt; mean) (Exp ≈ 0.63) |
|-----|-------------:|---------:|-------------:|-------------------------------:|
| 0–300 | 64 | 4.7 s | 0.80 | 0.52 |
| 900–1200 (peak) | 223 | 1.3 s | **0.93** | **0.62** |
| 1500–1800 | 96 | 3.1 s | 0.77 | 0.62 |

Gaps look **Poisson-like** (especially in the peak bin), not uniform clockwork —
good enough for “not a fixed movie,” but not a formal Poisson claim.

---

## Cross-file stochasticity

- Vehicle counts differ across seeds (742–849).
- Train and hold files are distinct draws (different OD tallies and departure
  times), not clones of one route movie.
- Example top OD pairs differ in rank/count between `fixed_1800` and `hold_00`,
  while still reflecting the same hub-heavy structure (e.g. heavy flow from
  eastern fringe stubs into central approaches).

Protocol implication: train on rotating `train_*`, score on `hold_*`
(`HELDOUT` / `HELDOUT_MEAN`). Do not compare raw throughput to old **3600 s**
tables — horizon and demand volume are both ~half of the classic movie setup.

---

## Verdict

| Claim | Status |
|-------|--------|
| Hub-centric OD | **Yes** (hub dest ~88%, hub-touch ~95%) |
| Fringe + within-network origins | **Yes** (~65 / ~35) |
| Shoulder + peak timeline | **Yes** (matches design weights) |
| Stochastic demand set + held-out | **Yes** |
| Literal Poisson flows end-to-end | **No** (random-in-bin; Poisson flag unused) |

If a paper needs explicit Poisson flows, regenerate with
`od2trips --flow-output.poisson` (or sample OD counts from Poisson rates) and
re-run this check.

---

## How to regenerate / re-check

```bash
python extras/gen_od_hub_demand_grid4x4.py
python extras/gen_od_hub_demand_grid4x4.py --validate-only
./extras/validate_od_hub_demand.sh
```

Docs: [`docs/OD_HUB_DEMAND.md`](../docs/OD_HUB_DEMAND.md).  
Generator: [`extras/gen_od_hub_demand_grid4x4.py`](../extras/gen_od_hub_demand_grid4x4.py).
