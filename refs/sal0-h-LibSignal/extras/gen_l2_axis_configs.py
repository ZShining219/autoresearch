#!/usr/bin/env python3
"""Write thin L2 single-axis agent configs under configs/tsc/.

Each file is hub OD + L1/L2 early-stop + one realism_full axis.
Does not submit jobs. Re-run anytime; files are deterministic.

    python extras/gen_l2_axis_configs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "tsc"

AGENTS = ("fixedtime", "maxpressure", "dqn", "presslight", "colight")
AXES = ("hetero", "slow_start", "crossing_proxy", "obs", "noise")
NETWORKS = (
    ("", "configs/tsc/od_hub_1800_base.yml"),
    ("_1x21", "configs/tsc/od_hub_1x21_1800_base.yml"),
)
BASELINES = {"fixedtime", "maxpressure"}
BASELINE_TRAINER = """
trainer:
  episodes: 1
  early_stop: False
"""


def render(agent: str, axis: str, suffix: str, demand: str) -> str:
    loc = "Level 2" if not suffix else "Level 2 (Ingolstadt)"
    lines = [
        f"# {loc} single-axis ablation: hub OD + {axis} only.",
        "# Demand/early-stop from the L1/L2 od_hub base; axis params from",
        f"# configs/tsc/axis_{axis}_world.yml (same values as realism_full).",
        "includes:",
        f"  - configs/tsc/{agent}.yml",
        f"  - configs/tsc/axis_{axis}_world.yml",
        f"  - {demand}",
    ]
    if agent in BASELINES:
        lines.append(BASELINE_TRAINER.rstrip("\n"))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    written = []
    for agent in AGENTS:
        for axis in AXES:
            for suffix, demand in NETWORKS:
                name = f"{agent}_odh_l2_{axis}{suffix}.yml"
                path = CFG / name
                path.write_text(
                    render(agent, axis, suffix, demand), encoding="utf-8"
                )
                written.append(path.relative_to(ROOT).as_posix())
    print(f"Wrote {len(written)} configs:")
    for rel in written:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
