"""Scenario registry: reusable local SUMO scenarios + generated grids.

net/rou paths resolved at call time; nothing here runs SUMO.
"""
import os

PROJ = "/data/users/zfh/workspace/projects"
AR = os.path.join(PROJ, "auto_research")

SCENARIOS = {
    # real 1x1 (Hangzhou bc-tyc), 8 green phases — smoke/mechanism tests
    "hz1x1": dict(
        net=f"{PROJ}/simOfflineTSC/data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/"
            f"hangzhou_1x1_bc-tyc_18041610_1h.net.xml",
        rou=f"{PROJ}/simOfflineTSC/data/raw_data/hangzhou_1x1_bc-tyc_18041610_1h/"
            f"hangzhou_1x1_bc-tyc_18041610_1h.rou.xml"),
    # RESCO grid4x4 benchmark (16 signals, 8 green phases, 1200 demand seeds)
    "resco4x4": dict(
        net=f"{PROJ}/dualight/sumo_files/scenarios/resco_envs/grid4x4/grid4x4.net.xml",
        rou_template=f"{PROJ}/dualight/sumo_files/scenarios/resco_envs/grid4x4/"
                     f"grid4x4_{{seed}}.rou.xml"),
    # real 4x4 (Hangzhou Gudang; LMP-GNN validation net)
    "gudang4x4": dict(
        net=f"{PROJ}/simOfflineTSC/data/raw_data/hangzhou_4x4_gudang_18041610_1h/"
            f"hangzhou_4x4_gudang_18041610_1h.net.xml",
        rou=f"{PROJ}/simOfflineTSC/data/raw_data/hangzhou_4x4_gudang_18041610_1h/"
            f"hangzhou_4x4_gudang_18041610_1h.rou.xml"),
}


def resolve(name, seed=None, rou=None):
    """Return (net_path, rou_path)."""
    s = SCENARIOS[name]
    net = s["net"]
    if rou is not None:
        return net, rou
    if "rou_template" in s:
        return net, s["rou_template"].format(seed=seed if seed is not None else 0)
    return net, s["rou"]


def gen_grid_scenario(rows, cols, out_dir, seed=0, rate=0.22, horizon=3600):
    """Generated grid (fills the 2x2 gap and enables demand sweeps)."""
    from .netgen import build_scenario
    return build_scenario(rows, cols, out_dir, seed=seed,
                          rate_per_od=rate, horizon=horizon)
