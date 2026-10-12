#!/usr/bin/env python3
"""
Hub-centric synthetic OD demand generator for Ingolstadt arterial (1800 s).

Owns sampling logic (gravity OD, hubs, fringe/internal mix, shoulder+peak
timeline). Uses SUMO od2trips + duarouter to emit a demand set:

  data/raw_data/ingolstadt21/od_hubs/demand_set/
    fixed_1800.rou.xml
    train_00..train_{N-1}.rou.xml
    hold_00..hold_{K-1}.rou.xml
    manifest.json

Usage:
    python extras/gen_ingolstadt21_taz.py   # once
    python extras/gen_od_hub_demand_ingolstadt21.py
    python extras/gen_od_hub_demand_ingolstadt21.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NET = REPO / "data/raw_data/ingolstadt21/ingolstadt21.net.xml"
TAZ = REPO / "data/raw_data/ingolstadt21/od_hubs/taz.xml"
OUT_DIR = REPO / "data/raw_data/ingolstadt21/od_hubs/demand_set"
REL_PREFIX = "raw_data/ingolstadt21/od_hubs/demand_set"

EPISODE_LEN = 1800
N_TRAIN = 10
N_HOLD = 3
# ~half of the afternoon-peak trip count (~4200 over 3600 s) for a 1800 s episode.
TARGET_VEHICLES = 2100
HUB_ID = "hub_cbd"
# ~65% fringe origins, ~35% internal (plan: 60–70 / 30–40)
FRINGE_ORIGIN_SHARE = 0.65
HUB_DEST_SHARE = 0.70  # of destinations that prefer hub
GRAVITY_BETA = 1.2
TRAIN_SEED_BASE = 2000
HOLD_SEED_BASE = 9000
FIXED_SEED = 42

# Shoulder + peak over 1800 s (6 × 300 s bins); weights normalized later
TIMELINE_BINS = [
    (0, 300, 0.08),
    (300, 600, 0.12),
    (600, 900, 0.22),  # rising
    (900, 1200, 0.28),  # peak
    (1200, 1500, 0.18),
    (1500, 1800, 0.12),
]


def _find_sumo_binary(name: str) -> str:
    env = os.environ.get("SUMO_HOME")
    if env:
        candidate = Path(env) / "bin" / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"Cannot find SUMO binary '{name}'. Set SUMO_HOME or PATH.")


def _load_taz(path: Path):
    root = ET.parse(path).getroot()
    zones = {}
    for taz in root.findall("taz"):
        tid = taz.get("id")
        sources = [s.get("id") for s in taz.findall("tazSource")]
        sinks = [s.get("id") for s in taz.findall("tazSink")]
        if not sources and taz.get("edges"):
            sources = sinks = taz.get("edges").split()
        zones[tid] = {"sources": sources, "sinks": sinks}
    return zones


def _zone_xy(net_path: Path, zones: dict):
    """Approximate zone centroid from edge from/to nodes in net.xml."""
    root = ET.parse(net_path).getroot()
    node_xy = {
        n.get("id"): (float(n.get("x")), float(n.get("y")))
        for n in root.findall("junction")
        if n.get("type") != "internal"
    }
    edge_mid = {}
    for e in root.findall("edge"):
        eid = e.get("id")
        if eid.startswith(":"):
            continue
        a, b = e.get("from"), e.get("to")
        if a in node_xy and b in node_xy:
            ax, ay = node_xy[a]
            bx, by = node_xy[b]
            edge_mid[eid] = ((ax + bx) / 2, (ay + by) / 2)
    centroids = {}
    for tid, z in zones.items():
        pts = [edge_mid[e] for e in (z["sources"] + z["sinks"]) if e in edge_mid]
        if not pts:
            centroids[tid] = (0.0, 0.0)
        else:
            centroids[tid] = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    return centroids


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) + 1.0


def _classify_zones(zones: dict):
    fringe = [z for z in zones if z.startswith("fringe_")]
    internal = [z for z in zones if z.startswith("internal_")]
    hubs = [z for z in zones if z.startswith("hub_")]
    return fringe, internal, hubs


def sample_od_matrix(zones, centroids, rng: random.Random, n_vehicles: int):
    """Sample OD counts with hub-centric gravity and fringe/internal origin mix."""
    fringe, internal, hubs = _classify_zones(zones)
    if not hubs:
        raise SystemExit("taz.xml must define at least one hub_* zone")

    origin_pool = []
    for z in fringe:
        origin_pool.extend([(z, "fringe")] * 10)
    for z in internal:
        origin_pool.extend([(z, "internal")] * 10)

    # Build destination preference: hub-heavy
    dest_weights = {}
    for z in zones:
        if z in hubs:
            dest_weights[z] = 5.0
        elif z.startswith("fringe_"):
            dest_weights[z] = 1.0
        else:
            dest_weights[z] = 0.4

    counts = defaultdict(int)
    n_fringe_o = 0
    n_internal_o = 0
    n_hub_touch = 0

    for _ in range(n_vehicles):
        if rng.random() < FRINGE_ORIGIN_SHARE and fringe:
            o = rng.choice(fringe)
            n_fringe_o += 1
        else:
            o = rng.choice(internal) if internal else rng.choice(fringe)
            n_internal_o += 1

        # Destination: with HUB_DEST_SHARE force hub, else gravity over all
        if rng.random() < HUB_DEST_SHARE:
            d = rng.choice(hubs)
        else:
            weights = []
            dests = []
            for d in zones:
                if d == o:
                    continue
                w = dest_weights[d] / (_dist(centroids[o], centroids[d]) ** GRAVITY_BETA)
                dests.append(d)
                weights.append(w)
            d = rng.choices(dests, weights=weights, k=1)[0]

        counts[(o, d)] += 1
        if o in hubs or d in hubs:
            n_hub_touch += 1

    return counts, {
        "n_vehicles": n_vehicles,
        "fringe_origin_frac": n_fringe_o / n_vehicles,
        "internal_origin_frac": n_internal_o / n_vehicles,
        "hub_touch_frac": n_hub_touch / n_vehicles,
    }


def _write_tazrelation(path: Path, counts: dict, begin: float, end: float):
    """Write OD for one time interval in tazRelation format."""
    root = ET.Element(
        "data",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "https://sumo.dlr.de/xsd/datamode_file.xsd",
        },
    )
    interval = ET.SubElement(
        root, "interval", id="pkw", begin=str(int(begin)), end=str(int(end))
    )
    for (o, d), c in sorted(counts.items()):
        if c <= 0:
            continue
        ET.SubElement(
            interval, "tazRelation", from_=o, to=d, count=str(int(c))
        )
        # ElementTree uses from_ kw; fix attribute name
    # Fix 'from_' -> 'from'
    xml = ET.tostring(root, encoding="unicode")
    xml = xml.replace("from_=", "from=")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml)


def _split_counts_by_timeline(counts: dict, rng: random.Random):
    """Split total OD counts across timeline bins by weight."""
    total_w = sum(w for _, _, w in TIMELINE_BINS)
    # Expand to list of trips then assign bins (preserves totals)
    trips = []
    for (o, d), c in counts.items():
        trips.extend([(o, d)] * int(c))
    rng.shuffle(trips)
    bin_counts = [defaultdict(int) for _ in TIMELINE_BINS]
    # cumulative assignment by weight
    cuts = []
    acc = 0.0
    n = len(trips)
    for i, (_, _, w) in enumerate(TIMELINE_BINS):
        acc += w / total_w
        cuts.append(int(round(acc * n)))
    prev = 0
    for i, cut in enumerate(cuts):
        for o, d in trips[prev:cut]:
            bin_counts[i][(o, d)] += 1
        prev = cut
    return bin_counts


def _write_vtype_routes_header():
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n'
        '  <vType id="pkw" length="5.0" width="2.0" minGap="2.5" '
        'maxSpeed="11.111" accel="2.0" decel="4.5"/>\n'
    )


def generate_one(seed: int, out_rou: Path, zones, centroids, od2trips, duarouter):
    rng = random.Random(seed)
    n_veh = max(1, int(rng.gauss(TARGET_VEHICLES, 25)))
    counts, stats = sample_od_matrix(zones, centroids, rng, n_veh)
    bin_counts = _split_counts_by_timeline(counts, rng)

    with tempfile.TemporaryDirectory(prefix="od_hub_") as tmp:
        tmp = Path(tmp)
        trip_parts = []
        for i, ((b, e, _), bc) in enumerate(zip(TIMELINE_BINS, bin_counts)):
            if sum(bc.values()) == 0:
                continue
            od_path = tmp / f"od_{i}.xml"
            # write tazRelation manually with correct 'from'
            root = ET.Element("data")
            interval = ET.SubElement(
                root, "interval", id="pkw", begin=str(b), end=str(e)
            )
            for (o, d), c in bc.items():
                el = ET.SubElement(interval, "tazRelation")
                el.set("from", o)
                el.set("to", d)
                el.set("count", str(int(c)))
            ET.ElementTree(root).write(od_path, encoding="utf-8", xml_declaration=True)

            trips_i = tmp / f"trips_{i}.xml"
            cmd = [
                od2trips,
                "-n", str(TAZ),
                "-z", str(od_path),
                "-o", str(trips_i),
                "-b", str(b),
                "-e", str(e),
                "--prefix", f"b{i}_",
                "--different-source-sink", "true",
            ]
            # random departures: do NOT pass --spread.uniform
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            trip_parts.append(trips_i)

        # Merge trips
        merged = tmp / "trips_merged.xml"
        vehicles = []
        for p in trip_parts:
            r = ET.parse(p).getroot()
            for tag in ("trip", "vehicle"):
                vehicles.extend(list(r.findall(tag)))
        mroot = ET.Element("routes")
        for v in vehicles:
            mroot.append(v)
        ET.ElementTree(mroot).write(merged, encoding="utf-8", xml_declaration=True)

        routed = tmp / "routed.rou.xml"
        cmd = [
            duarouter,
            "-n", str(NET),
            "-r", str(merged),
            "-o", str(routed),
            "--ignore-errors", "true",
            "--repair", "true",
            "--no-warnings", "true",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"duarouter failed: {proc.stderr[-2000:]}")

        # Normalize: ensure vType present, filter to vehicles with route
        rroot = ET.parse(routed).getroot()
        out_root = ET.Element(
            "routes",
            attrib={
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
            },
        )
        ET.SubElement(
            out_root,
            "vType",
            id="pkw",
            length="5.0",
            width="2.0",
            minGap="2.5",
            maxSpeed="11.111",
            accel="2.0",
            decel="4.5",
        )
        kept = 0
        for veh in rroot.findall("vehicle"):
            route = veh.find("route")
            if route is None or not route.get("edges"):
                continue
            veh.set("type", "pkw")
            # renumber
            veh.set("id", str(kept))
            out_root.append(veh)
            kept += 1
        out_rou.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(out_root).write(out_rou, encoding="utf-8", xml_declaration=True)
        stats["routed_vehicles"] = kept
        stats["seed"] = seed
        return stats


def validate_demand_set(manifest: dict) -> bool:
    """Print validation report; return True if checks pass."""
    ok = True
    print("=== OD hub demand validation ===")
    files = (
        [manifest["fixed"]]
        + manifest.get("train_set", [])
        + manifest.get("heldout", [])
    )
    hub_fracs = []
    fringe_fracs = []
    internal_fracs = []
    depart_bins_all = Counter()
    veh_counts = []

    zones = _load_taz(TAZ)
    fringe_edges = set()
    internal_edges = set()
    hub_edges = set()
    for zid, z in zones.items():
        edges = set(z["sources"] + z["sinks"])
        if zid.startswith("fringe_"):
            fringe_edges |= edges
        elif zid.startswith("internal_"):
            internal_edges |= edges
        elif zid.startswith("hub_"):
            hub_edges |= edges

    for rel in files:
        path = REPO / "data" / rel if not rel.startswith("/") else Path(rel)
        # paths in manifest are raw_data/... under data/
        path = REPO / "data" / rel
        if not path.exists():
            # try as written relative to data/
            path = REPO / rel
        if not path.exists():
            print(f"MISSING {rel}")
            ok = False
            continue
        root = ET.parse(path).getroot()
        vehs = list(root.findall("vehicle"))
        veh_counts.append(len(vehs))
        hub_touch = fringe_o = internal_o = 0
        for v in vehs:
            edges = v.find("route").get("edges").split()
            o, d = edges[0], edges[-1]
            if o in fringe_edges:
                fringe_o += 1
            elif o in internal_edges or o in hub_edges:
                # hub edges as origin count as within-network for mix reporting
                internal_o += 1
            if o in hub_edges or d in hub_edges:
                hub_touch += 1
            depart_bins_all[int(float(v.get("depart"))) // 300] += 1
        n = max(len(vehs), 1)
        hub_fracs.append(hub_touch / n)
        fringe_fracs.append(fringe_o / n)
        internal_fracs.append(internal_o / n)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"files={len(files)} veh_counts={veh_counts}")
    print(f"mean hub_touch_frac={mean(hub_fracs):.3f} (target >= 0.50)")
    print(f"mean fringe_origin_frac={mean(fringe_fracs):.3f} (target ~0.60-0.70)")
    print(f"mean within_origin_frac={mean(internal_fracs):.3f} (target ~0.30-0.40)")
    print(f"depart 300s bins={dict(sorted(depart_bins_all.items()))}")

    if mean(hub_fracs) < 0.50:
        print("FAIL hub share")
        ok = False
    if not (0.50 <= mean(fringe_fracs) <= 0.85):
        print("WARN fringe origin mix outside 0.50-0.85")
    if len(set(veh_counts)) < 2 and len(veh_counts) > 1:
        print("WARN vehicle counts identical across all files")
    # timeline non-flat: peak bin should exceed first bin
    if depart_bins_all:
        first = depart_bins_all.get(0, 0)
        peak = max(depart_bins_all.get(i, 0) for i in range(6))
        if peak <= first:
            print("WARN timeline may be flat (peak <= first bin)")
        else:
            print("OK non-flat timeline (peak > first bin)")
    # train vs hold disjoint seeds in manifest
    train_seeds = {s["seed"] for s in manifest.get("file_stats", []) if s.get("role") == "train"}
    hold_seeds = {s["seed"] for s in manifest.get("file_stats", []) if s.get("role") == "hold"}
    if train_seeds & hold_seeds:
        print("FAIL train/hold seed overlap")
        ok = False
    else:
        print("OK train/hold seeds disjoint")
    print("PASS" if ok else "FAIL")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-hold", type=int, default=N_HOLD)
    args = parser.parse_args()

    man_path = OUT_DIR / "manifest.json"
    if args.validate_only:
        if not man_path.exists():
            raise SystemExit(f"missing {man_path}")
        ok = validate_demand_set(json.loads(man_path.read_text()))
        raise SystemExit(0 if ok else 1)

    if not TAZ.exists():
        raise SystemExit(f"missing TAZ {TAZ}")
    if not NET.exists():
        raise SystemExit(f"missing net {NET}")

    od2trips = _find_sumo_binary("od2trips")
    duarouter = _find_sumo_binary("duarouter")
    zones = _load_taz(TAZ)
    centroids = _zone_xy(NET, zones)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file_stats = []
    # fixed
    fixed_path = OUT_DIR / "fixed_1800.rou.xml"
    print(f"generating fixed seed={FIXED_SEED}...")
    st = generate_one(FIXED_SEED, fixed_path, zones, centroids, od2trips, duarouter)
    st["role"] = "fixed"
    st["file"] = f"{REL_PREFIX}/fixed_1800.rou.xml"
    file_stats.append(st)
    print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    train_files = []
    for i in range(args.n_train):
        seed = TRAIN_SEED_BASE + i
        path = OUT_DIR / f"train_{i:02d}.rou.xml"
        print(f"generating train_{i:02d} seed={seed}...")
        st = generate_one(seed, path, zones, centroids, od2trips, duarouter)
        st["role"] = "train"
        st["file"] = f"{REL_PREFIX}/train_{i:02d}.rou.xml"
        file_stats.append(st)
        train_files.append(st["file"])
        print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    hold_files = []
    for i in range(args.n_hold):
        seed = HOLD_SEED_BASE + i
        path = OUT_DIR / f"hold_{i:02d}.rou.xml"
        print(f"generating hold_{i:02d} seed={seed}...")
        st = generate_one(seed, path, zones, centroids, od2trips, duarouter)
        st["role"] = "hold"
        st["file"] = f"{REL_PREFIX}/hold_{i:02d}.rou.xml"
        file_stats.append(st)
        hold_files.append(st["file"])
        print(f"  -> {st['routed_vehicles']} veh hub_touch={st['hub_touch_frac']:.2f}")

    manifest = {
        "network": "sumo1x21",
        "episode_len_s": EPISODE_LEN,
        "target_vehicles": TARGET_VEHICLES,
        "taz": "raw_data/ingolstadt21/od_hubs/taz.xml",
        "fixed": f"{REL_PREFIX}/fixed_1800.rou.xml",
        "train_set": train_files,
        "heldout": hold_files,
        "fringe_origin_share_target": FRINGE_ORIGIN_SHARE,
        "hub_dest_share_target": HUB_DEST_SHARE,
        "gravity_beta": GRAVITY_BETA,
        "timeline_bins_s": [[b, e, w] for b, e, w in TIMELINE_BINS],
        "routing": "duarouter_shortest_path",
        "departures": "od2trips_random",
        "file_stats": file_stats,
        "note": (
            "Hub-centric synthetic OD on Ingolstadt arterial; mixed fringe/internal "
            "origins; 1800s shoulder+peak; demand set + held-out protocol."
        ),
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path}")
    validate_demand_set(manifest)


if __name__ == "__main__":
    main()
