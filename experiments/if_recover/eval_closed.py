#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Closed-loop eval: MaxPressure x {clean, masked, masked+IF} on RobustLight Hangzhou 4x4.

Runs under the py310 env (cityflow + torch2). Replicates utils/model_test.py
semantics for MODEL_NAME=MaxPressure:
  - mask applied to converted feature xs[0] ([16,12] traffic_movement_pressure_queue)
    at direction dims given by NOISE_DIRECTION via index_maps
  - optional recovery via IFRecoverer.demask_state (same call signature)
  - recovered/masked feature written back into next_state at step_num>=4
  - controller consumes (possibly recovered) state next step
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROOT = "/data/users/zfh/workspace/projects/auto_research"
sys.path.insert(0, os.path.join(ROOT, "refs", "RobustLight"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.cityflow_env import CityFlowEnv  # noqa: E402
from ifrec.recoverer import IFRecoverer  # noqa: E402

DATA_DIR = os.path.join(ROOT, "refs", "RobustLight", "data", "Hangzhou", "4_4")

CONF = {
    "LIST_MODEL_NEED_TO_UPDATE": [],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": "MaxPressure",
    "TOP_K_ADJACENCY": 5,
    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 16,
    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,
    "BINARY_PHASE_EXPANSION": True,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],
    "INTERVAL": 1,
    "LIST_STATE_FEATURE": ["traffic_movement_pressure_queue", "cur_phase"],
    "DIC_REWARD_INFO": {"pressure": 0},
    "PHASE": {1: [0, 1, 0, 1, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0, 1, 0, 1],
              3: [1, 0, 1, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 1, 0, 1, 0]},
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
    "NUM_AGENTS": 16,
    "NUM_GENERATORS": 1,
    "NUM_ROW": 4,
    "NUM_COL": 4,
    "ROADNET_FILE": "roadnet_4_4.json",
    "index_maps": {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]},
}


class MPController:
    """Rule-based MaxPressure, verbatim from models/maxpressure_agent.choose_action."""

    def __init__(self):
        self.action = None

    def choose(self, s):
        if s["cur_phase"][0] == -1:
            return self.action
        p = np.asarray(s["traffic_movement_pressure_queue"], dtype=np.float64)
        self.action = int(np.argmax([p[1] + p[4], p[7] + p[10], p[0] + p[3], p[6] + p[9]]))
        return self.action


def convert(state):
    """Same as MaxPressureAgent.convert_state_to_input for the single feature."""
    return np.array([s["traffic_movement_pressure_queue"] for s in state],
                    dtype=np.float32)  # [16,12]


def make_mask(directions):
    mask = np.ones((16, 12), dtype=np.float32)
    dims = []
    for d in directions:
        dims += CONF["index_maps"][d]
    mask[:, dims] = 0.0
    return mask


def run_episode(flow_path, work_dir, arm, directions, ckpt=None, dump=False,
                fill_c=0.0, autoreg=False):
    conf = dict(CONF)
    conf["TRAFFIC_FILE"] = os.path.basename(flow_path)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    log_dir = os.path.join(work_dir, "test_round", "round_0")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(flow_path, os.path.join(work_dir, os.path.basename(flow_path)))
    shutil.copy(os.path.join(DATA_DIR, "roadnet_4_4.json"),
                os.path.join(work_dir, "roadnet_4_4.json"))
    env = CityFlowEnv(path_to_log=log_dir, path_to_work_directory=work_dir,
                      dic_traffic_env_conf=conf)
    mps = [MPController() for _ in range(16)]
    rec = IFRecoverer(ckpt, autoreg=autoreg) if arm == "if" else None
    mask = make_mask(directions) if directions else None

    state = env.reset()
    max_steps = int(conf["RUN_COUNTS"] / conf["MIN_ACTION_TIME"])
    step_num, done = 0, False
    mae_masked_vals = []
    dump = {"pq_true": [], "pq_used": [], "phase": [], "action": []} if dump else None

    while not done and step_num < max_steps:
        action_list = [mps[i].choose(state[i]) for i in range(16)]
        next_state, reward, done, _ = env.step(action_list)
        if dump is not None:
            dump["action"].append(list(action_list))
        xs = convert(next_state)
        if mask is not None:
            x_masked = mask * xs
            if rec is not None:
                aoh = np.eye(4, dtype=np.float32)[np.asarray(action_list)]
                xs_used = rec.demask_state(x_masked, aoh, None, mask)
            elif arm == "fillc":
                xs_used = x_masked + (1.0 - mask) * fill_c
            else:
                xs_used = x_masked
            if step_num >= 4:
                for i, s in enumerate(next_state):
                    s["traffic_movement_pressure_queue"] = xs_used[i].tolist()
            miss = mask < 0.5
            mae_masked_vals.append(np.abs(xs_used - xs)[miss].mean())
            if dump is not None:
                dump["pq_true"].append(xs.copy())
                dump["pq_used"].append(xs_used.copy())
                dump["phase"].append([s["cur_phase"][0] for s in next_state])
        state = next_state
        step_num += 1

    run_counts = conf["RUN_COUNTS"]
    veh_dur, veh_left = {}, set()
    for inter in env.list_intersection:
        for vid, r in inter.get_dic_vehicle_arrive_leave_time().items():
            lt = r["leave_time"]
            if lt == lt:
                veh_left.add(vid)
            else:
                lt = run_counts
            veh_dur[vid] = veh_dur.get(vid, 0.0) + (lt - r["enter_time"])
    durs = np.array(list(veh_dur.values()))
    res = {"att": float(durs.mean()) if len(durs) else float("nan"),
           "n_in": len(veh_dur), "n_out": len(veh_left), "steps": step_num,
           "mae_masked": float(np.mean(mae_masked_vals)) if mae_masked_vals else None}
    if dump is not None:
        dpath = os.path.join(work_dir, "dump.npz")
        np.savez_compressed(dpath, **{k: np.array(v) for k, v in dump.items()})
        res["dump"] = dpath
    env.end_cityflow()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True)
    ap.add_argument("--arm", choices=["clean", "masked", "if", "fillc"], required=True)
    ap.add_argument("--fillc", type=float, default=0.0)
    ap.add_argument("--dirs", default="W", help="comma list e.g. W or W,E")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results/if_recover/model_if.pt"))
    ap.add_argument("--autoreg", action="store_true",
                    help="feed recovered values back into window history (use with --selfcond ckpt)")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "data/if_recover/rundirs/eval"))
    ap.add_argument("--dump", action="store_true", help="dump true/used pq per step for offline analysis")
    args = ap.parse_args()
    dirs = [d for d in args.dirs.split(",") if d]
    if args.arm == "clean":
        dirs = []
    res = run_episode(os.path.abspath(args.flow), args.workdir, args.arm,
                      dirs, args.ckpt, dump=args.dump, fill_c=args.fillc,
                      autoreg=args.autoreg)
    res.update({"arm": args.arm, "dirs": dirs, "flow": os.path.basename(args.flow)})
    print(json.dumps(res))


if __name__ == "__main__":
    main()
