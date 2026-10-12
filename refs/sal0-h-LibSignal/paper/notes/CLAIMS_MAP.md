# Claims map (repo-backed)

Status legend: **supported** (numbers in logs/notebooks) · **partial** (some nets / sync gaps) · **planned** · **open** (research question).

| # | Claim | Evidence | Status |
|---|--------|----------|--------|
| C1 | LibSignal Grid4×4 SUMO table is reproducible as a gate | `realism_benchmark_4x4.ipynb` §1 vs paper Table 8 | supported (pre-TLS-fix net) |
| C2 | Pre-fix grid TLS was invalid for realism science | `docs/SIGNAL_CONTROL_THEORY.md`; notebook §1–2 | supported |
| C3 | On corrected homo 4×4, RL ≈ MP ≪ FixedTime | Table homo: DQN 167.9, MP 173.3, FT 219.1 | supported |
| C4 | Five realism axes are composable (plant + sensor) | `docs/REALISM_FULL.md`, axis docs, configs `*_realism_full` | supported |
| C5 | Single axes add modest ATT tax; noise hurts CoLight | `realism_benchmark_4x4.ipynb` axis tables | supported |
| C6 | Under `realism_full`, MP is the anchor; CoLight collapses | MP 237.5 vs CoLight 494.6; PressLight +10.5 vs MP | supported |
| C7 | OD-hub L1: RL can match MP on held-out demand | L1 held-out ~157s for DQN/CoLight/MP | supported |
| C8 | OD-hub L2: compound realism preferentially hurts some RL | DQN +88, CoLight +279 vs L1; MP +30 | supported |
| C9 | Mean ATT hides skew but may not reorder homo 4×4 | `improved_metrics.ipynb` | supported |
| C10 | Idle/travel effectiveness is a useful secondary signal | median waiting frac 0.05 vs 0.22 FT | **open** (promote?) |
| C11 | Rankings are topology-dependent (4×4 vs 1×21) | `ingolstadt21.ipynb` §6 commentary + homo ranks | partial |
| C12 | Classical MP does **not** uniquely fail under realism | C6–C8 + discussion | supported on 4×4; partial on 1×21 |
| C13 | Effect sizes of RL vs MP are often small vs realism uplift | homo Δ few seconds vs realism_full +64s MP | supported |
| C14 | Ghost physics is a ceiling baseline | `physics_mode: ghost`, TRAINING_GUIDE | implemented; numbers TBD in paper tables |
| C15 | Actor–critic (IPPO/MADDPG) under same protocol | agents exist in repo | planned |
| C16 | LLMLight under same protocol | lit only so far | planned |
| C17 | Multi-seed statistical significance | mostly seed 42 | planned |

## Notebooks (read-only sources)

- `result_analysis/realism_benchmark_4x4.ipynb`
- `result_analysis/od_hub_benchmark_4x4.ipynb`
- `result_analysis/improved_metrics.ipynb`
- `result_analysis/ingolstadt21.ipynb`
- `result_analysis/adaptive_episodes.ipynb`
- `result_analysis/od_hub_demand_distribution.md`
