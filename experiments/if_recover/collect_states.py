#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect complete MaxPressure state trajectories from RobustLight's CityFlow env.

Runs under the `dualight` conda env (py3.6 + cityflow). Mirrors utils/model_test.py's
control loop (NOISE off) and dumps per-decision-step arrays to npz:

    pq     [T+1, 16, 12]  traffic_movement_pressure_queue (the masked feature)
    phase  [T+1, 16]      cur_phase index per intersection
    action [T,   16]      action chosen at each step (phase index 0..3)
    att, n_in, n_out     episode travel-time metrics (summary.py semantics)
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

ROBUST_DIR = "/data/users/zfh/workspace/projects/auto_research/refs/RobustLight"
sys.path.insert(0, ROBUST_DIR)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

if not hasattr(nn, "Mish"):  # torch 1.5 lacks Mish; needed by Diffusion_Predictor ctor
    class Mish(nn.Module):
        def forward(self, x):
            return x * torch.tanh(F.softplus(x))
    nn.Mish = Mish

from utils import config  # noqa: E402
from utils.config import Robust_config  # noqa: E402
from utils.cityflow_env import CityFlowEnv  # noqa: E402
from models.maxpressure_agent import MaxPressureAgent  # noqa: E402

DATA_DIR = os.path.join(ROBUST_DIR, "data", "Hangzhou", "4_4")


def build_conf(traffic_file):
    num = 16
    dic_traffic_env_conf = dict(config.dic_traffic_env_conf)
    dic_traffic_env_conf.update({
        "NUM_AGENTS": num,
        "NUM_INTERSECTIONS": num,
        "NUM_GENERATORS": 1,
        "MODEL_NAME": "MaxPressure",
        "RUN_COUNTS": 3600,
        "NUM_ROW": 4,
        "NUM_COL": 4,
        "TRAFFIC_FILE": os.path.basename(traffic_file),
        "ROADNET_FILE": "roadnet_4_4.json",
        "LIST_STATE_FEATURE": ["traffic_movement_pressure_queue", "cur_phase"],
        "DIC_REWARD_INFO": {"pressure": 0},
        "NUM_ROUNDS": 1,
        "is_test": True,
        "log_writer": False,
        "inference_epoch": 0,
        "is_inference": False,
        "sota_path": "model/MaxPressure/" + os.path.basename(traffic_file)[:-5],
        "NOISE_SCALE": 3.5,
        "NOISE_TYPE": 0,
        "NOISE_LEVEL": 0,
        "device": "cpu",
        "index_maps": {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]},
        "NOISE_DIRECTION": [],
        "inference_config": Robust_config,
    })
    dic_agent_conf = dict(config.DIC_BASE_AGENT_CONF)
    dic_agent_conf.update({"FIXED_TIME": [15, 15, 15, 15]})
    return dic_traffic_env_conf, dic_agent_conf


def run_episode(traffic_file, work_dir, out_npz):
    dic_traffic_env_conf, dic_agent_conf = build_conf(traffic_file)
    log_dir = os.path.join(work_dir, "test_round", "round_0")
    os.makedirs(log_dir, exist_ok=True)
    # CityFlow resolves files relative to work_dir
    flow_abs = traffic_file if os.path.isabs(traffic_file) else os.path.abspath(traffic_file)
    if not os.path.exists(flow_abs):
        flow_abs = os.path.join(DATA_DIR, traffic_file)
    for f in [flow_abs, os.path.join(DATA_DIR, "roadnet_4_4.json")]:
        shutil.copy(f, os.path.join(work_dir, os.path.basename(f)))

    dic_path = {"PATH_TO_MODEL": work_dir, "PATH_TO_WORK_DIRECTORY": work_dir}
    env = CityFlowEnv(path_to_log=log_dir,
                      path_to_work_directory=work_dir,
                      dic_traffic_env_conf=dic_traffic_env_conf)
    agents = [MaxPressureAgent(dic_agent_conf=dic_agent_conf,
                               dic_traffic_env_conf=dic_traffic_env_conf,
                               dic_path=dic_path, cnt_round=0,
                               intersection_id=str(i))
              for i in range(dic_traffic_env_conf["NUM_AGENTS"])]

    state = env.reset()
    n_agents = dic_traffic_env_conf["NUM_AGENTS"]
    max_steps = int(dic_traffic_env_conf["RUN_COUNTS"] / dic_traffic_env_conf["MIN_ACTION_TIME"])

    pq = [agents[0].convert_state_to_input(state)[0].copy()]
    phase = [[s["cur_phase"][0] for s in state]]
    actions = []

    step_num = 0
    done = False
    while not done and step_num < max_steps:
        action_list = []
        for i in range(n_agents):
            action_list.append(agents[i].choose_action(True, state[i]))
        next_state, reward, done, _ = env.step(action_list)
        actions.append(list(action_list))
        pq.append(agents[0].convert_state_to_input(next_state)[0].copy())
        phase.append([s["cur_phase"][0] for s in next_state])
        state = next_state
        step_num += 1

    # episode ATT, summary.py semantics
    run_counts = dic_traffic_env_conf["RUN_COUNTS"]
    veh_dur = {}
    veh_left = set()
    for inter in env.list_intersection:
        for vid, rec in inter.get_dic_vehicle_arrive_leave_time().items():
            lt = rec["leave_time"]
            if lt == lt:  # not NaN
                veh_left.add(vid)
            else:
                lt = run_counts
            veh_dur[vid] = veh_dur.get(vid, 0.0) + (lt - rec["enter_time"])
    durs = np.array(list(veh_dur.values()))
    att = float(durs.mean()) if len(durs) else float("nan")
    n_in = len(veh_dur)
    n_out = len(veh_left)

    np.savez_compressed(out_npz,
                        pq=np.array(pq, dtype=np.float32),
                        phase=np.array(phase, dtype=np.int64),
                        action=np.array(actions, dtype=np.int64),
                        att=att, n_in=n_in, n_out=n_out,
                        traffic_file=os.path.basename(traffic_file))
    print("saved {0}: steps={1} att={2:.2f} in={3} out={4}".format(out_npz, step_num, att, n_in, n_out))
    env.end_cityflow()
    return att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, help="flow json path or basename under data/Hangzhou/4_4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    os.makedirs(args.workdir)
    run_episode(args.flow, args.workdir, args.out)


if __name__ == "__main__":
    main()
