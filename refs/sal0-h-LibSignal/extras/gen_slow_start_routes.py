#!/usr/bin/env python3
"""
Generate route files without inline <vType> for slow-start experiments.
The vType is provided by vTypes_slow_start.add.xml via --additional-files.

Vehicles without explicit type="pkw" will use the default vType, which is
the pkw defined in the additional file (loaded before route files).

Usage:
    python extras/gen_slow_start_routes.py
"""
import os
import xml.etree.ElementTree as ET

NETWORKS = [
    {
        "name": "sumo1x1",
        "src": "data/raw_data/cologne1/cologne1.rou.xml",
        "dst": "data/raw_data/cologne1/cologne1_slow_start.rou.xml",
    },
    {
        "name": "sumo4x4",
        "src": "data/raw_data/grid4x4/grid4x4.rou.xml",
        "dst": "data/raw_data/grid4x4/grid4x4_slow_start.rou.xml",
    },
    {
        "name": "sumo7x28",
        "src": "data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml",
        "dst": "data/raw_data/manhattan_28x7/manhattan_28x7_slow_start.rou.xml",
    },
]


def gen_slow_start(src_path, dst_path):
    """Remove inline <vType> and set type='pkw' on all vehicles so the
    vType from vTypes_slow_start.add.xml (with tau=1.8) is actually used."""
    tree = ET.parse(src_path)
    root = tree.getroot()

    removed = 0
    for vtype in root.findall("vType"):
        root.remove(vtype)
        removed += 1

    # Ensure every vehicle explicitly references type="pkw"
    # (without this, SUMO uses its built-in default vType, ignoring our tau override)
    for veh in root.findall("vehicle"):
        if "type" not in veh.attrib:
            veh.set("type", "pkw")

    tree.write(dst_path, encoding="utf-8", xml_declaration=True)
    vehicles = root.findall("vehicle")
    print(f"[{os.path.basename(dst_path)}] {removed} vType(s) removed, {len(vehicles)} vehicles tagged type='pkw'")
    return len(vehicles)


def main():
    print("Generating slow-start route files (vType removed, loaded via additional-files)...")
    for net in NETWORKS:
        src = net["src"]
        dst = net["dst"]
        if not os.path.exists(src):
            print(f"  SKIP {net['name']}: {src} not found")
            continue
        gen_slow_start(src, dst)
    print("Done.")


if __name__ == "__main__":
    main()
