#!/usr/bin/env python3
"""
Generate crossing_proxy_lanes.json for a SUMO network.

NEMA concurrent walk (docs/CROSSING_PROXY.md), two lane groups per through phase:
  - through_incoming: parallel through approaches on green (driver yield / corner delay)
  - conflict_lanes: crosswalk-street movements with G/g/s (turns / permissive conflicts)

Runtime also applies actuated phase extension (hold green) on ped call.

Usage:
    export SUMO_HOME=/path/to/sumo
    python extras/gen_crossing_proxy_lanes.py --network grid4x4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

THROUGH_PHASES = ("0", "4")

THROUGH_PHASE_NOTES = {
    "0": "NS through + E-W walk: N/S through_incoming + E-W conflict (G/g/s)",
    "4": "EW through + N-S walk: E/W through_incoming + N-S conflict (G/g/s)",
}

NETWORKS = {
    "grid4x4": {
        "net": REPO / "data/raw_data/grid4x4/grid4x4.net.xml",
        "rou": REPO / "data/raw_data/grid4x4/grid4x4.rou.xml",
        "out": REPO / "data/raw_data/grid4x4/crossing_proxy_lanes.json",
        "through_incoming_dirs": {
            "0": ("N", "S"),
            "4": ("E", "W"),
        },
        "conflict_axis": {
            "0": "EW",
            "4": "NS",
        },
    },
}


def _cardinal(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx))
    if ang >= 45.0:
        return "N"
    if ang >= -45.0:
        return "E"
    if ang >= -135.0:
        return "S"
    return "W"


def _approach_direction(lane_id: str, cx: float, cy: float, lane_shape_fn) -> str:
    shape = lane_shape_fn(lane_id)
    fx, fy = shape[0]
    return _cardinal(cx - fx, cy - fy)


def _movement_axis(from_lane: str, cx: float, cy: float, lane_shape_fn) -> str:
    return _approach_direction(from_lane, cx, cy, lane_shape_fn)


def _green_phase_states(tls_id: str, eng) -> list[str]:
    logic = eng.trafficlight.getAllProgramLogics(tls_id)[0]
    states = []
    for phase in logic.phases:
        state = phase.state
        if "y" in state:
            continue
        if state.count("r") + state.count("s") == len(state):
            continue
        states.append(state)
    return states


def build_lane_map(net_path: Path, rou_path: Path, spec: dict) -> dict:
    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError(
            "SUMO_HOME must be set (e.g. export SUMO_HOME=/opt/homebrew/opt/sumo/share/sumo)"
        )
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)

    import sumolib
    import traci

    net = sumolib.net.readNet(str(net_path))
    through_incoming_dirs = spec["through_incoming_dirs"]
    conflict_axis = spec["conflict_axis"]
    sumo_bin = sumolib.checkBinary("sumo")
    traci.start(
        [
            sumo_bin,
            "-n",
            str(net_path),
            "-r",
            str(rou_path),
            "--no-warnings",
            "true",
            "--end",
            "1",
        ]
    )

    intersections = {}
    try:
        for tls_id in traci.trafficlight.getIDList():
            node = net.getNode(tls_id)
            cx, cy = node.getCoord()
            links = traci.trafficlight.getControlledLinks(tls_id)
            green_states = _green_phase_states(tls_id, traci)

            by_dir = {"N": set(), "E": set(), "S": set(), "W": set()}
            for link in links:
                if not link:
                    continue
                from_lane = link[0][0]
                if from_lane.startswith(":"):
                    continue
                by_dir[_approach_direction(from_lane, cx, cy, traci.lane.getShape)].add(
                    from_lane
                )

            phase_map = {}
            for phase_key in THROUGH_PHASES:
                idx = int(phase_key)
                through_dirs = through_incoming_dirs[phase_key]
                through_incoming = sorted(
                    lane for d in through_dirs for lane in by_dir[d]
                )
                conflict = set()
                if idx < len(green_states):
                    state = green_states[idx]
                    cross = conflict_axis[phase_key]
                    for link_idx, link in enumerate(links):
                        if not link or link_idx >= len(state):
                            break
                        if state[link_idx] not in ("G", "g", "s"):
                            continue
                        from_lane = link[0][0]
                        if from_lane.startswith(":"):
                            continue
                        if _movement_axis(from_lane, cx, cy, traci.lane.getShape) == cross:
                            conflict.add(from_lane)
                phase_map[phase_key] = {
                    "through_incoming": through_incoming,
                    "conflict_lanes": sorted(conflict),
                }
            intersections[tls_id] = phase_map
    finally:
        traci.close()

    return {
        "network": net_path.stem.replace(".net", ""),
        "mechanism": "actuated_extension_plus_through_yield_and_conflicts",
        "through_phases": list(THROUGH_PHASES),
        "phase_semantics": THROUGH_PHASE_NOTES,
        "intersections": intersections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        default="grid4x4",
        choices=list(NETWORKS.keys()),
    )
    args = parser.parse_args()
    spec = NETWORKS[args.network]
    data = build_lane_map(spec["net"], spec["rou"], spec)
    spec["out"].parent.mkdir(parents=True, exist_ok=True)
    with open(spec["out"], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    sample = data["intersections"][next(iter(data["intersections"]))]["0"]
    print(
        f"Wrote {spec['out']} — sample phase0 "
        f"through={len(sample['through_incoming'])} "
        f"conflict={len(sample['conflict_lanes'])}"
    )


if __name__ == "__main__":
    main()
