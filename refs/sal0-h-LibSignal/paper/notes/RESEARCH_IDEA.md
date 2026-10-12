# Research idea — organized outline

## One-sentence thesis
**Leaderboards on idealized TSC sims overstate RL gains; under realism-composed plant, sensor, and demand stress, classical max-pressure remains competitive and rankings become fragile.**

## Story spine
1. Reproduce LibSignal → establish competence / find TLS bug → correct network.
2. Define 5 realism axes (hetero, slow-start, crossing proxy, penetration, noise) + OD-hub demand.
3. Ablate axes → compose `realism_full` → show MP anchor / RL fragility.
4. Move to Ingolstadt 1×21 → show topology reshuffles ranks.
5. Critique metrics (ATT mean vs distributions / idle share) and small effect sizes.
6. Ghost ceiling; future actor–critic + LLMLight under the *same* protocol.

## What is *not* the contribution
- A new SOTA RL algorithm.
- Full multimodal ped simulation.
- Proven real-world deployment.

## Contribution type
**Benchmarking / evaluation methodology + empirical study** on an extended LibSignal fork.
