"""Deterministic SUMO grid network + demand generation.

Generates a rows x cols signalised grid with 3-lane approaches and a 4-phase
plan per intersection. The tlLogic embedded in the net is only a placeholder;
the runtime controller redefines the program (LibSignal convention).

Coordinate convention (netgenerate-style): node g_i_j at (j*L, i*L),
i = row index (north increases with i), j = column index (east increases with j).
Edges are named e{i}{j}_N/S/E/W pointing at the neighbour in that direction.
"""
import os
import subprocess
import shutil
import numpy as np
import xml.etree.ElementTree as ET


def _sumo_bin(name):
    p = shutil.which(name)
    if p:
        return p
    import sumo as _sumo_pkg
    cand = os.path.join(os.path.dirname(_sumo_pkg.__file__), "bin", name)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"SUMO binary {name} not found")

LANES = 3
SPEED = 13.89          # m/s (~50 km/h)
LENGTH = 300.0         # block length between intersections
ARM_LEN = 60.0         # stub length on network boundary


def _node_id(i, j):
    return f"g{i}_{j}"


def gen_grid(rows, cols, out_dir, arm_len=ARM_LEN, block_len=LENGTH, lanes=LANES):
    """Write nod/edg files and run netconvert. Returns path to net.xml."""
    os.makedirs(out_dir, exist_ok=True)
    nodes, edges = [], []
    # interior signal nodes
    for i in range(rows):
        for j in range(cols):
            nodes.append(f'<node id="{_node_id(i,j)}" x="{j*block_len}" y="{i*block_len}" type="traffic_light"/>')
    # boundary stub nodes + edges
    dirs = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}
    in_edges, out_edges = {}, {}
    for i in range(rows):
        for j in range(cols):
            for d, (di, dj) in dirs.items():
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    a, b = _node_id(i, j), _node_id(ni, nj)
                    edges.append((f"e{i}_{j}_{d}", a, b, block_len))
                else:
                    stub = f"b{i}_{j}_{d}"
                    sx, sy = (j + dj) * block_len, (i + di) * block_len
                    # boundary node sits one stub-length outside the signal
                    sx = j * block_len + dj * arm_len
                    sy = i * block_len + di * arm_len
                    nodes.append(f'<node id="{stub}" x="{sx}" y="{sy}" type="priority"/>')
                    edges.append((f"e{i}_{j}_{d}", _node_id(i, j), stub, arm_len))
                    edges.append((f"e{i}_{j}_{d}_in", stub, _node_id(i, j), arm_len))
    nod_path = os.path.join(out_dir, "grid.nod.xml")
    edg_path = os.path.join(out_dir, "grid.edg.xml")
    with open(nod_path, "w") as f:
        f.write("<nodes>\n" + "\n".join(nodes) + "\n</nodes>\n")
    with open(edg_path, "w") as f:
        f.write("<edges>\n")
        for eid, a, b, ln in edges:
            f.write(f'<edge id="{eid}" from="{a}" to="{b}" numLanes="{lanes}" speed="{SPEED}" length="{ln}"/>\n')
        f.write("</edges>\n")
    net_path = os.path.join(out_dir, "grid.net.xml")
    subprocess.run([_sumo_bin("netconvert"), "-n", nod_path, "-e", edg_path, "-o", net_path,
                    "--no-turnarounds", "true", "--tls.default-type", "static",
                    "--junctions.corner-detail", "0"],
                   check=True, capture_output=True)
    return net_path


def gen_routes(net_path, out_path, horizon=3600, rate_per_od=0.22, seed=0, rng_extra=None):
    """Uniform border->border flows with seeded jitter.

    Every boundary stub edge sources flow to every opposite-side stub edge.
    rate_per_od ~ vehicles/second per OD pair baseline. For a 4x4 grid there are
    4*(rows+cols) boundary stubs; pair each inbound stub with a few exits to keep
    ODs sparse and demand tunable.
    """
    rng = np.random.default_rng(seed)
    import sumolib
    net = sumolib.net.readNet(net_path)
    border_in, border_out = [], []
    for edge in net.getEdges():
        fr, to = edge.getFromNode(), edge.getToNode()
        if to.getType() in ("priority", "dead_end") and fr.getType() == "traffic_light":
            border_out.append(edge.getID())
        if fr.getType() in ("priority", "dead_end") and to.getType() == "traffic_light":
            border_in.append(edge.getID())
    vehs, vid = [], 0
    for ein in border_in:
        dests = [e for e in border_out if not e.startswith(ein.split("_in")[0])]
        # keep through + turns: pick 3 exits per entrance deterministically
        dests = sorted(dests)
        rng.shuffle(dests)
        for eout in dests[:3]:
            rate = rate_per_od * (0.7 + 0.6 * rng.random())
            vehs.append(f'<flow id="f{vid}" from="{ein}" to="{eout}" begin="0" end="{horizon}" '
                        f'vehsPerHour="{rate*3600:.1f}" departLane="best" departSpeed="max"/>')
            vid += 1
    with open(out_path, "w") as f:
        f.write('<routes>\n<vType id="car" accel="2.6" decel="4.5" length="5" '
                'minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>\n')
        f.write("\n".join(vehs) + "\n</routes>\n")
    return out_path


def gen_sumocfg(net_path, rou_path, out_path):
    with open(out_path, "w") as f:
        f.write("<configuration>\n<input>\n"
                f'<net-file value="{os.path.basename(net_path)}"/>\n'
                f'<route-files value="{os.path.basename(rou_path)}"/>\n'
                "</input>\n</configuration>\n")
    return out_path


def build_scenario(rows, cols, out_dir, seed=0, rate_per_od=0.22, horizon=3600):
    """Create net + route + sumocfg under out_dir. Returns dict of paths."""
    os.makedirs(out_dir, exist_ok=True)
    net = gen_grid(rows, cols, out_dir)
    rou = gen_routes(net, os.path.join(out_dir, "grid.rou.xml"),
                     horizon=horizon, rate_per_od=rate_per_od, seed=seed)
    cfg = gen_sumocfg(net, rou, os.path.join(out_dir, "grid.sumocfg"))
    return {"net": net, "rou": rou, "cfg": cfg}
