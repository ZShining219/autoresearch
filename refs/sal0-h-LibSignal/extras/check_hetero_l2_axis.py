#!/usr/bin/env python3
"""Did the post-805dc4b L2-axis hetero jobs actually mix the fleet?

Run on gpujobs (before download) or on the Mac after rsync.

Exit 0 only if all 10 journal cells (5 agents x 2 nets) are a real mixed-fleet
run, not the Sep 12 no-op (FixedTime ATT glued at 186.6263 / 280.8920).

Usage:
  python extras/check_hetero_l2_axis.py
  python extras/check_hetero_l2_axis.py --tsc data/output_data/tsc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("fixedtime", "maxpressure", "dqn", "presslight", "colight")
NETS = (("", "sumo4x4"), ("_1x21", "sumo1x21"))
# Pre-rewrite FixedTime held-out means (trucks never spawned).
NOOP_ATT = {"sumo4x4": 186.6263, "sumo1x21": 280.8920}
NOOP_TOL = 0.05
# rewrite_route_file_hetero_mix landed 2026-09-14 08:57 UTC.
REWRITE_PREFIX = "2026_09_14"


def hold_mean(logger: Path, stem: str):
    vals = []
    for i in range(3):
        p = logger / f"{stem}_{i:02d}_meta.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        v = meta.get("mean_travel_time_s") or meta.get("avg_travel_time_metric")
        if v is not None:
            vals.append(float(v))
    if len(vals) == 3:
        return sum(vals) / 3.0, vals
    return None, vals


def newest_run_stamp(logger: Path) -> str:
    stamps = []
    for p in logger.glob("*_BRF.log"):
        stamps.append(p.name.split("_BRF.log")[0])
    for p in logger.glob("*_DTL.log"):
        stamps.append(p.name.split("_DTL.log")[0])
    return max(stamps) if stamps else ""


def fleet_hits(paths) -> list[str]:
    hits = []
    pat = re.compile(r"\[Hetero\] mixed fleet|hetero_mix")
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if pat.search(text):
            hits.append(str(path))
    return hits


def slurm_logs(repo: Path, cfg: str, net: str):
    logs = repo / "logs"
    if not logs.is_dir():
        return []
    out = []
    for p in sorted(logs.glob("l2ax_*hetero*.out")):
        if cfg in p.name:
            out.append(p)
    if out:
        return out
    for p in sorted(logs.glob("*hetero*.out")):
        if cfg in p.name:
            out.append(p)
    return out


def check_cell(repo: Path, tsc: Path, agent: str, suffix: str, net: str) -> dict:
    cfg = f"{agent}_odh_l2_hetero{suffix}"
    logger = tsc / f"sumo_{cfg}" / net / "l2_axis_hetero" / "logger"
    row = {
        "agent": agent,
        "network": net,
        "config": cfg,
        "status": "MISSING",
        "stamp": "",
        "att": None,
        "source": "",
        "fleet": False,
        "detail": "",
    }
    if not logger.is_dir():
        row["detail"] = "no logger dir"
        return row

    stem = "new_metrics_hold" if agent in ("fixedtime", "maxpressure") else "new_metrics_best_hold"
    att, vals = hold_mean(logger, stem)
    if att is None:
        att, vals = hold_mean(logger, "new_metrics_hold")
        stem = "new_metrics_hold"
    row["att"] = att
    row["source"] = stem if att is not None else "incomplete"
    row["stamp"] = newest_run_stamp(logger)

    hits = fleet_hits([*logger.glob("*.log"), *slurm_logs(repo, cfg, net)])
    row["fleet"] = bool(hits)

    if att is None:
        row["status"] = "INCOMPLETE"
        row["detail"] = f"holds={len(vals)} stamp={row['stamp'] or 'none'}"
        return row

    stale = (not row["stamp"]) or (row["stamp"] < REWRITE_PREFIX)
    noop = agent == "fixedtime" and abs(att - NOOP_ATT[net]) < NOOP_TOL
    if stale:
        row["status"] = "STALE"
        row["detail"] = (
            f"newest log {row['stamp'] or 'none'} is before {REWRITE_PREFIX} "
            f"(pre-rewrite no-op). ATT={att:.4f}"
        )
        return row
    if noop:
        row["status"] = "NOOP"
        row["detail"] = (
            f"FixedTime ATT {att:.4f} still matches no-op {NOOP_ATT[net]:.4f}"
        )
        return row
    if agent == "fixedtime" and not row["fleet"]:
        # ATT moved, so rewrite almost certainly ran; slurm .out may not be local.
        row["status"] = "OK"
        row["detail"] = (
            f"ATT={att:.4f} moved off no-op {NOOP_ATT[net]:.4f}; "
            "no [Hetero] line in local files (ok on Mac after rsync)"
        )
        return row
    row["status"] = "OK"
    row["detail"] = f"ATT={att:.4f} fleet={row['fleet']}"
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tsc", type=Path, default=None)
    args = p.parse_args()
    tsc = args.tsc or (ROOT / "data/output_data/tsc")

    print(f"Checking L2-axis hetero under {tsc}")
    print(f"Need a run stamp >= {REWRITE_PREFIX} and FixedTime ATT off the no-op.")
    print("")
    fmt = "  {:<12} {:<8} {:<10} {:>10}  {}"
    print(fmt.format("AGENT", "NET", "STATUS", "ATT", "DETAIL"))
    rows = []
    for agent in AGENTS:
        for suffix, net in NETS:
            rows.append(check_cell(ROOT, tsc, agent, suffix, net))
    ok = 0
    for r in rows:
        att = f"{r['att']:.4f}" if r["att"] is not None else "NA"
        print(fmt.format(r["agent"], r["network"], r["status"], att, r["detail"]))
        if r["status"] == "OK":
            ok += 1

    print("")
    print(f"{ok}/10 cells OK")
    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        print("Not ready to push. On gpujobs look for:")
        print("  grep -E '\\[Hetero\\] mixed fleet' logs/l2ax_*hetero*.out")
        print("  squeue -u $USER")
        print("Sep 12 loggers are the no-op. Keep waiting or resubmit:")
        print("  ./extras/submit_l2_axes.sh hetero")
        return 1

    print("Hetero rewrite ran. On the Mac:")
    print("  AXES=hetero ./extras/download_l2_axes.sh")
    print("  python extras/check_hetero_l2_axis.py")
    print("  git add data/output_data/tsc/sumo_{fixedtime,maxpressure,dqn,presslight,colight}_odh_l2_hetero*")
    print("  git commit && git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
