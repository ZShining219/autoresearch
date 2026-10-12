#!/usr/bin/env python3
"""Draft MPLight signal_config from a live LibSignal World (cologne3 rotation recipe).

Run from repo root (traffic conda env):
  python extras/dump_mplight_signal_config.py --network sumo1x21
  python extras/dump_mplight_signal_config.py --network sumo_cologne3 --validate

Outputs valid_acts + lane_order keyed by SUMO TL id in intersection_ids order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PHASE_PAIRS = [
    [1, 7], [2, 8], [1, 2], [7, 8],
    [4, 10], [5, 11], [10, 11], [4, 5],
]
PHASE_NAMES = [
    "NT_ST", "NL_SL", "NT_NL", "ST_SL",
    "ET_WT", "EL_WL", "WT_WL", "ET_EL",
]
CANON = {
    "NR": 0, "NT": 1, "NL": 2,
    "ER": 3, "ET": 4, "EL": 5,
    "SR": 6, "ST": 7, "SL": 8,
    "WR": 9, "WT": 10, "WL": 11,
}
CANON_APPROACH = {idx: name[0] for name, idx in CANON.items()}
# cologne3 corridor rotation: geo compass -> canonical approach label
ROTATE = {"E": "N", "S": "E", "W": "S", "N": "W"}

HAND_COLOGNE3 = {
    "valid_acts": {
        "cluster_2415878664_254486231_359566_359576": {0: 0, 1: 1, 4: 2, 5: 3},
        "360086": {0: 0, 1: 1, 4: 2, 5: 3},
        "360082": {0: 0, 1: 1, 7: 2},
    },
    "lane_order": {
        "360082": {1: 0, 2: 1, 4: 2, 7: 3, 8: 4},
        "360086": {1: 0, 2: 1, 4: 2, 7: 3, 8: 4, 10: 5},
        "cluster_2415878664_254486231_359566_359576": {
            1: 0, 2: 1, 4: 2, 5: 3, 7: 4, 8: 5, 10: 6, 11: 7,
        },
    },
}


def _compass(angle: float) -> str:
    deg = math.degrees(angle) % 360
    if deg < 45 or deg >= 315:
        return "N"
    if 45 <= deg < 135:
        return "E"
    if 135 <= deg < 225:
        return "S"
    return "W"


def _load_conn(net_path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    out: dict[tuple[str, int], tuple[str, str]] = {}
    for c in ET.parse(net_path).getroot().findall("connection"):
        tls = c.get("tl")
        if not tls:
            continue
        fl = f"{c.get('from')}_{c.get('fromLane')}"
        out[(tls, int(c.get("linkIndex")))] = (c.get("dir", "s"), fl)
    return out


def _obs_lanes(inter) -> list[str]:
    return [lane for lane in inter.lanes if lane[:-2] in inter.in_roads]


def _road_compass(inter) -> dict[str, str]:
    out = {}
    for road in inter.in_roads:
        idx = inter.roads.index(road)
        out[road] = _compass(inter.directions[idx])
    return out


def _canonical_approach(geo: str) -> str:
    return ROTATE.get(geo, geo)


def _lane_movements(tls_id: str, inter, conn_dir) -> dict[str, str]:
    """Map inbound lane id -> movement T/L (rights skipped) on its canonical approach."""
    per_lane_dirs: dict[str, set[str]] = {}
    for (tls, idx), (d, fl) in conn_dir.items():
        if tls != tls_id:
            continue
        if fl[:-2] not in inter.in_roads:
            continue
        per_lane_dirs.setdefault(fl, set()).add(d)

    by_road: dict[str, list[str]] = {}
    for lane in _obs_lanes(inter):
        by_road.setdefault(lane[:-2], []).append(lane)

    result: dict[str, str] = {}
    for road, lanes in by_road.items():
        lanes = sorted(lanes, key=lambda l: int(l.rsplit("_", 1)[-1]))
        if len(lanes) == 1:
            lane = lanes[0]
            dirs = per_lane_dirs.get(lane, {"s"})
            if "r" in dirs and "s" not in dirs and "t" not in dirs and "l" not in dirs:
                continue
            # single-lane arm: through if any t/s (even when l also exists)
            result[lane] = "T" if ("s" in dirs or "t" in dirs) else "L"
            continue
        t_assigned = False
        for lane in lanes:
            dirs = per_lane_dirs.get(lane, {"s"})
            if "r" in dirs and "s" not in dirs and "t" not in dirs and "l" not in dirs:
                continue
            if "l" in dirs and t_assigned:
                result[lane] = "L"
            elif not t_assigned and ("s" in dirs or "t" in dirs):
                result[lane] = "T"
                t_assigned = True
            elif "l" in dirs:
                result[lane] = "L"
            elif not t_assigned:
                result[lane] = "T"
                t_assigned = True
            else:
                result[lane] = "L"
    return result


def _lane_order(inter, lane_mov) -> dict[int, int]:
    road_geo = _road_compass(inter)
    order: dict[int, int] = {}
    for obs_i, lane in enumerate(_obs_lanes(inter)):
        if lane not in lane_mov:
            continue
        road = lane[:-2]
        app = _canonical_approach(road_geo[road])
        mov = lane_mov[lane]
        canon = CANON[app + mov]
        if canon not in order:
            order[canon] = obs_i
    return order


def _link_canonical(direction: str, from_lane: str, inter, road_geo) -> int | None:
    """Canonical movement of a single controlled link.

    Phase signatures must be read per link, not per lane: a shared lane carries
    both through and left, and only the link's own ``dir`` tells the two apart.
    Rights are dropped because no ``phase_pairs`` entry references them, and 't'
    (U-turn) is dropped because it is not a through movement -- counting it as
    one makes left-turn phases masquerade as through phases.
    """
    road = from_lane[:-2]
    if road not in inter.in_roads:
        return None
    if direction == "s":
        mov = "T"
    elif direction in ("l", "L"):
        mov = "L"
    else:
        return None
    return CANON.get(_canonical_approach(road_geo[road]) + mov)


def _approaches_present(inter, road_geo) -> set[str]:
    return {_canonical_approach(road_geo[road]) for road in inter.in_roads}


def _green_canonicals(inter, lane_mov, conn_dir) -> list[set[int]]:
    """Canonical movements released by each local green phase.

    Protected greens ('G'/'s') are preferred; a phase holding only permissive
    greens ('g') falls back to those so it still gets a movement signature.
    """
    tls = inter.id
    links = {idx: (d, fl) for (t, idx), (d, fl) in conn_dir.items() if t == tls}
    road_geo = _road_compass(inter)
    greens: list[set[int]] = []
    for phase in inter.green_phases:
        strict: set[int] = set()
        permissive: set[int] = set()
        for i, ch in enumerate(phase.state):
            if ch not in ("G", "s", "g"):
                continue
            if i not in links:
                continue
            canon = _link_canonical(*links[i], inter, road_geo)
            if canon is None:
                continue
            (strict if ch in ("G", "s") else permissive).add(canon)
        greens.append(strict or permissive)
    return greens


def _available_movements(inter, conn_dir) -> set[int]:
    """Canonical movements that physically exist at this junction."""
    road_geo = _road_compass(inter)
    avail = set()
    for (t, _), (d, fl) in conn_dir.items():
        if t != inter.id:
            continue
        canon = _link_canonical(d, fl, inter, road_geo)
        if canon is not None:
            avail.add(canon)
    return avail


def _pair_score(active: set[int], avail: set[int], approaches: set[str], pair) -> int:
    """Rank how well a canonical phase pair describes a local green.

    Tiering matters for irregular junctions: a pair whose missing movement does
    not exist at all (T-junction with no south leg) is a far better label than
    one whose missing movement exists but is deliberately held red. The approach
    bonus then prefers pairs that stay on legs the junction actually has, which
    is what keeps a 3-leg node on ET_EL instead of the phantom EL_WL.
    """
    need = set(pair)
    hit = len(active & need)
    if not hit:
        return 0
    missing = need - active
    if not missing:
        score = 300
    elif not missing & avail:
        score = 200
    else:
        score = 100
    if all(CANON_APPROACH[m] in approaches for m in need):
        score += 50
    return score + hit


def _valid_acts(inter, lane_mov, conn_dir) -> dict[int, int]:
    """Map canonical phase index -> local green index.

    Every local green gets its own canonical pair, so the returned values are
    exactly 0..n-1 (required: MPLight builds ``reverse_valid`` by inverting this
    dict and indexes it with explorer slots 0..len-1) and no phase is orphaned.
    """
    greens = _green_canonicals(inter, lane_mov, conn_dir)
    avail = _available_movements(inter, conn_dir)
    approaches = _approaches_present(inter, _road_compass(inter))
    n = len(greens)
    if n == 0:
        return {}
    if n > len(PHASE_PAIRS):
        raise ValueError(f"{inter.id}: {n} greens exceeds {len(PHASE_PAIRS)} phase pairs")

    scores = [
        [_pair_score(active, avail, approaches, pair) for pair in PHASE_PAIRS]
        for active in greens
    ]

    # n <= 8 and pairs == 8, so exhaustive search over injective assignments is cheap.
    best_choice, best_score = None, None
    for choice in itertools.permutations(range(len(PHASE_PAIRS)), n):
        total = sum(scores[pi][pair] for pi, pair in enumerate(choice))
        if best_score is None or total > best_score:
            best_score, best_choice = total, choice

    return {pair_idx: pi for pi, pair_idx in sorted(
        enumerate(best_choice), key=lambda kv: kv[1])}


def _boot_world(network: str):
    from common.registry import Registry
    from common import interface
    from utils.logger import build_config
    from world.world_sumo import World

    args = Namespace(
        thread_num=4, ngpu="-1", prefix="dump", seed=42, debug=False,
        interface="libsumo", delay_type="apx", task="tsc", agent="mplight",
        world="sumo", network=network, dataset="onfly",
    )
    config, _ = build_config(args)
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.Logger_path_Interface(config)
    interface.World_param_Interface(config)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)
    out = Path(Registry.mapping["logger_mapping"]["path"].path) / "debug"
    os.makedirs(out, exist_ok=True)
    world = World(f"configs/sim/{network}.cfg", interface="libsumo")
    net = REPO / Registry.mapping["world_mapping"]["setting"].param["dir"]
    net = net / Registry.mapping["world_mapping"]["setting"].param["roadnetFile"]
    return world, _load_conn(net)


def _yaml_tls_id(tls_id: str) -> str:
    return tls_id[3:] if tls_id.startswith("GS_") else tls_id


def build(network: str) -> dict:
    world, conn_dir = _boot_world(network)
    valid_acts: dict[str, dict[int, int]] = {}
    lane_order: dict[str, dict[int, int]] = {}
    try:
        for tls_id in world.intersection_ids:
            inter = world.id2intersection[tls_id]
            lane_mov = _lane_movements(tls_id, inter, conn_dir)
            key = _yaml_tls_id(tls_id)
            lane_order[key] = _lane_order(inter, lane_mov)
            valid_acts[key] = _valid_acts(inter, lane_mov, conn_dir)
    finally:
        world.eng.close()
    return {"valid_acts": valid_acts, "lane_order": lane_order}


def _print_yaml_block(name: str, cfg: dict) -> None:
    print(f"    {name}: {{")
    print("      phase_pairs: [[1, 7], [2, 8], [1, 2], [7, 8], [4, 10], [5, 11], [10, 11], [4, 5]],")
    print("      valid_acts: {")
    last = len(cfg["valid_acts"]) - 1
    for i, tls_id in enumerate(cfg["valid_acts"]):
        acts = cfg["valid_acts"][tls_id]
        names = [PHASE_NAMES[k] for k in sorted(acts, key=acts.get)]
        comma = "" if i == last else ","
        print(f"        # {names}")
        print(f"        '{tls_id}': {acts}{comma}")
    print("      },")
    print("      lane_order: {")
    last = len(cfg["lane_order"]) - 1
    for i, tls_id in enumerate(cfg["lane_order"]):
        comma = "" if i == last else ","
        print(f"        '{tls_id}': {cfg['lane_order'][tls_id]}{comma}")
    print("      }")
    print("    }")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="sumo1x21")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--yaml", action="store_true")
    args = parser.parse_args()

    if args.validate:
        ref = build("sumo_cologne3")
        ok = True
        for tls_id, hand in HAND_COLOGNE3["valid_acts"].items():
            got = ref["valid_acts"].get(tls_id, {})
            if got != hand:
                print(f"MISMATCH valid_acts {tls_id}: got {got} want {hand}")
                ok = False
        for tls_id, hand in HAND_COLOGNE3["lane_order"].items():
            got = ref["lane_order"].get(tls_id, {})
            if got != hand:
                print(f"MISMATCH lane_order {tls_id}: got {got} want {hand}")
                ok = False
        print("cologne3 validate:", "PASS" if ok else "FAIL")
        if not ok:
            return 1

    cfg = build(args.network)
    net_name = json.loads((REPO / f"configs/sim/{args.network}.cfg").read_text()).get(
        "network", args.network
    )
    if args.yaml:
        _print_yaml_block(net_name, cfg)
        return 0
    for tls_id in cfg["valid_acts"]:
        acts = cfg["valid_acts"][tls_id]
        print(f"{tls_id}: valid_acts={acts}  # {[PHASE_NAMES[k] for k in acts]}")
        print(f"         lane_order={cfg['lane_order'][tls_id]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
