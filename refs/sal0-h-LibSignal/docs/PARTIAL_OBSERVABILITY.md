# Partial / Noisy Observability

A realism axis for the realism-decomposed RL-TSC benchmark (after `hetero` fleet-mix and
slow-start). Real controllers never see every car: they read loops, cameras, connected
vehicles, and nav-app probes — incomplete, noisy, uneven coverage. This module corrupts
**what the signal policy perceives** while leaving the SUMO **physics unchanged**, so the
controller "decides with bad information." The current phase is always known.

Two **composable** axes, both living in `world/world_sumo.py`, both a **no-op at their
default** (exact baseline), and both applying the **same seeded corruption for every agent**
so ground-truth ATT stays comparable across DQN / PressLight / CoLight / MaxPressure:

| Axis | Param (`world:`) | Default | Mechanism |
|---|---|---|---|
| Penetration rate | `obs_penetration` | `1.0` | each vehicle is visible w.p. `p`, persistent per trip |
| Gaussian count noise | `obs_count_noise_std` (+ `obs_noise_mode`) | `0.0` / `additive` | zero-mean Gaussian on per-lane counts, per step |

Ground-truth **ATT/throughput** are computed from a separate path (`self.vehicles`) and are
**never** corrupted — they remain the honest comparison metric. Logged **queue/delay/reward**
become *observed* (corrupted) quantities, which is the intended semantics (what the
controller's sensors report). `--seed`/`world.seed` seed the corruption identically to SUMO
and the RNGs (single unified seed; `--seed` overrides `world.seed`).

## Literature grounding

- **Chen, Fang & Sadeh (2022)**, *The Real Deal: A Review of Challenges and Opportunities in
  Moving RL-Based Traffic Signal Control Systems Towards Reality*, ATT '22 Workshop on Agents
  in Traffic and Transportation, Vienna. CEUR-WS Vol. 3173, ISSN 1613-0073. — §3 ("Uncertainty
  in detection") is exactly this axis; §3.2 notes connected-vehicle **penetration remains low**
  ([49]); §3.3 cites Gaussian noise on queue length ([59]) and detector failure ([58], [61]).
- **Tan, Sharma & Sarkar (2020)**, *Robust Deep Reinforcement Learning for Traffic Signal
  Control*, J. Big Data Analytics in Transportation 2(3):263–274,
  doi:10.1007/s42421-020-00029-6 (ref [59] above). — inject a **discrete zero-mean Gaussian
  into the queue-length state** (per element, floored at 0); our `additive` mode reproduces it.

---

## Axis 1 — Penetration rate (`obs_penetration`)

Each vehicle is independently **visible/connected with probability `p`, persistent for its
whole trip** (a car either carries the tech or it doesn't), via a seeded per-vehicle hash
`frac = md5(seed:veh_id)[:4]/2^32; visible iff frac < p`. Lane stats are built from visible
vehicles only (one added condition in `Intersection._get_vehicles`). `p=1.0` short-circuits to
"all visible" (exact baseline, no draws). `E[observed_count] ≈ p·true` (multiplicative
under-count; pressure, a difference of counts, stays directionally meaningful).

## Axis 2 — Gaussian count noise (`obs_count_noise_std`, `obs_noise_mode`)

Each per-lane count is corrupted by zero-mean Gaussian measurement noise, **re-sampled every
step**, rounded, clamped `>= 0` (`World._apply_obs_count_noise`, a single per-step post-pass in
`_update_infos`, so counts / queue / waiting-count / both pressures inherit one consistent
realization). `additive`: `observed = true + N(0, σ²)`; `proportional`:
`observed = true + N(0, (σ·true)²)`. `σ=0` is the exact baseline. Faithful to Tan et al.
(2020); `additive` is their mechanism, `proportional` is our extension. (A ≥0 clamp gives a
tiny positive bias only when true counts are near 0 — same as the source.)

Both axes compose: penetration first reduces counts to the visible subset, then Gaussian noise
perturbs them.

## Discarded axis — Bernoulli lane dropout

A third axis (whole-lane sensor dropout with a fill value) was prototyped and **discarded**.
Reason: a dead sensor emits *absence of data*, not a value — "fill 0" conflates the failure
with a naive controller policy that treats silence as "empty," so it models a specific
worst-case (starvation) rather than a faithful missing-sensor. The two axes above cover the
detection-uncertainty challenge cleanly. (History on branch `spike/obs-lane-dropout`, SHA
`75fc260`, if ever needed.)

---

## How to run

```bash
# baseline (full observability, exact counts)
python run.py --task tsc --agent maxpressure --world sumo --network sumo1x1 --interface libsumo --ngpu -1
# 10% penetration
python run.py --task tsc --agent maxpressure_obs   --world sumo --network sumo1x1 --interface libsumo --ngpu -1
# sigma=2 additive count noise
python run.py --task tsc --agent maxpressure_noise --world sumo --network sumo1x1 --interface libsumo --ngpu -1
```
Example configs ship for both axes: `{maxpressure,fixedtime,dqn}_obs.yml` (penetration) and
`{maxpressure,fixedtime,dqn}_noise.yml` (noise). Sweep by setting `world.obs_penetration` /
`world.obs_count_noise_std` in a config that includes the agent's yml; set both to combine.

## Behaviour (indicative; formal results tracked in #17)

- **MaxPressure** (observation-driven) degrades as realism rises. sumo1x1: penetration
  ATT 39→154 (throughput 1997→243 at p=0.05); noise ATT 39→54 (gentler — unbiased noise
  partly cancels in the pressure difference). sumo4x4 penetration ATT 173→571.
- **FixedTime** (ignores observations) has **invariant ground-truth ATT** across every setting
  — confirming physics is unchanged and ATT is honest ground truth; its *observed* queue still
  shifts with the corruption.

## Verification (mechanisms confirmed by instrumentation)

- **Penetration.** Cumulative visible fraction over the full vehicle population tracks `p`
  (p=0.1→0.104, 0.5→0.488, 0.9→0.891, 1.0→1.000); **zero persistence violations** (a vehicle's
  visibility never flips across its trip); true active-vehicle count is independent of `p`
  (physics untouched — the cars are all there, just uncounted).
- **Gaussian.** Isolating the noise at non-trivial counts gives mean(observed−true)≈0.00 and
  std≈σ (e.g. σ=2 → 2.02); the small positive bias at tiny counts is exactly the documented ≥0
  clamp. Noise is re-sampled each step (flickers) and is deterministic given (seed, time)
  (identical run-to-run on deterministic nets).
