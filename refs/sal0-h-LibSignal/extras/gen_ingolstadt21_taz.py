#!/usr/bin/env python3
"""
Build hub-centric TAZ zones for Ingolstadt arterial (ingolstadt21).

Zones:
  - fringe_{N,E,S,W}: dead-end boundary edges by compass quadrant
  - hub_cbd: approaches of the central TLS cluster (median corridor)
  - internal_{NW,NE,SW,SE}: remaining non-fringe edges near TLS, by quadrant

Usage:
    python extras/gen_ingolstadt21_taz.py
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NET = REPO / "data/raw_data/ingolstadt21/ingolstadt21.net.xml"
OUT = REPO / "data/raw_data/ingolstadt21/od_hubs/taz.xml"


def _cardinal(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx))
    if ang >= 45.0:
        return "N"
    if ang >= -45.0:
        return "E"
    if ang >= -135.0:
        return "S"
    return "W"


def build_taz(net_path: Path) -> ET.Element:
    root = ET.parse(net_path).getroot()
    junc = {
        j.get("id"): j
        for j in root.findall("junction")
        if j.get("type") != "internal"
    }
    tls_ids = {t.get("id") for t in root.findall("tlLogic")}
    tls_xy = []
    for tid in tls_ids:
        j = junc.get(tid)
        if j is None:
            continue
        tls_xy.append((float(j.get("x")), float(j.get("y"))))
    if not tls_xy:
        raise SystemExit("no TLS junctions found")
    cx = sum(p[0] for p in tls_xy) / len(tls_xy)
    cy = sum(p[1] for p in tls_xy) / len(tls_xy)
    # Hub: TLS closest to network centroid (central corridor cluster)
    hub_tls = sorted(
        tls_ids,
        key=lambda tid: (
            math.hypot(
                float(junc[tid].get("x")) - cx,
                float(junc[tid].get("y")) - cy,
            )
            if tid in junc
            else 1e18
        ),
    )[:5]

    fringe_src = defaultdict(list)
    fringe_sink = defaultdict(list)
    hub_edges = set()
    internal = defaultdict(list)

    for e in root.findall("edge"):
        eid = e.get("id")
        if eid is None or eid.startswith(":"):
            continue
        fr, to = e.get("from"), e.get("to")
        fj, tj = junc.get(fr), junc.get(to)
        if fj is None or tj is None:
            continue
        fx, fy = float(fj.get("x")), float(fj.get("y"))
        tx, ty = float(tj.get("x")), float(tj.get("y"))
        mx, my = (fx + tx) / 2.0, (fy + ty) / 2.0

        if fj.get("type") == "dead_end":
            fringe_src[_cardinal(mx - cx, my - cy)].append(eid)
        if tj.get("type") == "dead_end":
            fringe_sink[_cardinal(mx - cx, my - cy)].append(eid)

        if fr in hub_tls or to in hub_tls:
            hub_edges.add(eid)
            continue

        # Approaches touching any TLS, not fringe endpoints, become internal.
        if fr in tls_ids or to in tls_ids:
            if fj.get("type") == "dead_end" or tj.get("type") == "dead_end":
                continue
            quad = ("N" if my >= cy else "S") + ("E" if mx >= cx else "W")
            internal[quad].append(eid)

    taz_root = ET.Element("tazs")

    def add_taz(tid: str, sources, sinks):
        sources = sorted(set(sources))
        sinks = sorted(set(sinks))
        if not sources and not sinks:
            return
        taz = ET.SubElement(taz_root, "taz", id=tid)
        for s in sources:
            ET.SubElement(taz, "tazSource", id=s, weight="1.0")
        for s in sinks:
            ET.SubElement(taz, "tazSink", id=s, weight="1.0")
        print(f"  {tid}: sources={len(sources)} sinks={len(sinks)}")

    # Prefer readable names matching grid4x4 style
    name_map = {"N": "north", "E": "east", "S": "south", "W": "west"}
    for d, name in name_map.items():
        add_taz(f"fringe_{name}", fringe_src[d], fringe_sink[d])

    add_taz("hub_cbd", sorted(hub_edges), sorted(hub_edges))

    for quad, edges in sorted(internal.items()):
        add_taz(f"internal_{quad}", edges, edges)

    return taz_root


def main():
    print(f"Building TAZ from {NET}")
    taz = build_taz(NET)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(taz).write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
