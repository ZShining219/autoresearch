#!/usr/bin/env python3
"""
Generate routes_hetero.rou.xml for each network from its baseline route file.

Strategy (fair ablation):
  - Keep total demand (vehicle count + depart times) identical to baseline.
  - Remove the inline <vType> from the route file (vTypes come from
    vTypes_mixed.add.xml via --additional-files).
  - Assign type="car" to ~80% and type="truck" to ~20% of vehicles using a
    deterministic stride so the mix is reproducible across seeds.

Usage:
    python extras/gen_hetero_routes.py
"""
import os
import re
import xml.etree.ElementTree as ET

CAR_RATIO = 0.8   # 80% cars
TRUCK_RATIO = 0.2  # 20% trucks

NETWORKS = [
    {
        "name": "sumo1x1",
        "src": "data/raw_data/cologne1/cologne1.rou.xml",
        "dst": "data/raw_data/cologne1/cologne1_hetero.rou.xml",
    },
    {
        "name": "sumo4x4",
        "src": "data/raw_data/grid4x4/grid4x4.rou.xml",
        "dst": "data/raw_data/grid4x4/grid4x4_hetero.rou.xml",
    },
    {
        "name": "sumo7x28",
        "src": "data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml",
        "dst": "data/raw_data/manhattan_28x7/manhattan_28x7_hetero.rou.xml",
    },
]


def gen_hetero(src_path, dst_path):
    """Parse src route XML, drop inline vType, assign car/truck types, write dst."""
    tree = ET.parse(src_path)
    root = tree.getroot()

    # Remove all <vType> elements — they'll come from vTypes_mixed.add.xml
    for vtype in root.findall("vType"):
        root.remove(vtype)

    vehicles = root.findall("vehicle")
    n = len(vehicles)
    n_trucks = int(round(n * TRUCK_RATIO))
    n_cars = n - n_trucks

    # Deterministic assignment: every k-th vehicle is a truck, where k = n/n_trucks.
    # This spreads trucks evenly across the demand profile.
    if n_trucks > 0:
        stride = n / n_trucks
        truck_indices = set(int(i * stride) for i in range(n_trucks))
    else:
        truck_indices = set()

    for idx, veh in enumerate(vehicles):
        if idx in truck_indices:
            veh.set("type", "truck")
        else:
            veh.set("type", "car")

    # Write with XML declaration
    tree.write(dst_path, encoding="utf-8", xml_declaration=True)
    print(f"[{os.path.basename(dst_path)}] {n} vehicles -> {n_cars} cars, {n_trucks} trucks")
    return n, n_cars, n_trucks


def main():
    print("Generating hetero route files (80% car / 20% truck)...")
    for net in NETWORKS:
        src = net["src"]
        dst = net["dst"]
        if not os.path.exists(src):
            print(f"  SKIP {net['name']}: {src} not found")
            continue
        gen_hetero(src, dst)
    print("Done.")


if __name__ == "__main__":
    main()
