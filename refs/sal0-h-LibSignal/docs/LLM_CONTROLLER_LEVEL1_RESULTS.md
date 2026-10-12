# Level-1 LLM controller performance

These are the completed local LLM evaluations retained for the Level-1
performance report. Each run used one test episode over the three held-out
routes (`hold_00`, `hold_01`, and `hold_02`), 1,800 SUMO steps, and an action
interval of 10. Mean ATT is the `HELDOUT_MEAN` travel-time value recorded by
the simulator logger.

| Model | GPU | Runtime | Mean ATT |
| --- | --- | ---: | ---: |
| Traffic-R1 | H200 2g.35gb | 01:42:18 | 153.96990591855823 |
| Qwen2.5-7B-Instruct | H200 2g.35gb | 01:40:31 | 168.63806409430182 |
| Qwen3-4B no-thinking | H200 1g.18gb | 04:27:55 | 208.18235248221322 |
| Qwen3-4B thinking/1024 sampled | H200 2g.35gb | 06:24:22 | 165.29152813175708 |
| Qwen3.6-27B no-thinking | H200 7g.141gb | 06:03:48 | 239.51055791053344 |

The corresponding final BRF/DTL files are stored in the matching
`data/output_data/tsc/sumo_<model>/sumo4x4/<run>/logger/` directories. Raw
SLURM stdout/stderr remain outside version control in the local experiment
archive.
