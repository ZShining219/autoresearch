#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train RobustLight RL controllers (EfficientMPLight / EfficientColight) on CPU.

Replicates run_mplight.py / run_colight.py but with is_test=False (train mode),
device="cpu", and configurable rounds/flow. Run under the `dualight` conda env
(torch 1.5 + tensorflow 1.11 + cityflow).

Must be executed with cwd = refs/RobustLight (PATH_TO_DATA is relative).

Produces:
  model/<memo>/<traffic_file>_<ts>/round_{r}_inter_{i}.h5   (controller ckpts)
  records/<memo>/<traffic_file>_<ts>/...                    (logs + traffic_env.conf)
"""
import argparse
import os
import sys
import time

import numpy as np
import random
import torch

# torch>=1.9 APIs missing in dualight's torch 1.5; shim before RL imports.
if not hasattr(torch.nn, "Mish"):
    class _Mish(torch.nn.Module):
        def forward(self, x):
            return x * torch.tanh(torch.nn.functional.softplus(x))
    torch.nn.Mish = _Mish

# network_agent uses standalone-keras API (K.tile) absent from tf.keras 1.11.
from tensorflow.keras import backend as _K  # noqa: E402
import tensorflow as _tf  # noqa: E402
if not hasattr(_K, "tile"):
    _K.tile = _tf.tile

ROOT = "/data/users/zfh/workspace/projects/auto_research"
RL = os.path.join(ROOT, "refs", "RobustLight")
sys.path.insert(0, RL)
os.chdir(RL)  # PATH_TO_DATA etc are relative

from utils.utils import pipeline_wrapper, merge  # noqa: E402
from utils import config  # noqa: E402
from utils.config import Robust_config  # noqa: E402
from inferences.rl_dynamic_state_filling import Diffusion_Predictor  # noqa: E402

# network_agent.py:60-62 unconditionally warm-starts from sota ckpts that do
# not exist for fresh training; skip only when the target is missing.
_orig_load_weights = _tf.keras.Model.load_weights
def _safe_load_weights(self, filepath, **kw):
    if not os.path.exists(filepath):
        print("[ctl_train] skip missing sota weights:", filepath)
        return
    return _orig_load_weights(self, filepath, **kw)
_tf.keras.Model.load_weights = _safe_load_weights

_orig_inf_load = Diffusion_Predictor.load_model
def _safe_inf_load(self, model_dir, device_idx=0):
    if not os.path.isdir(model_dir):
        print("[ctl_train] skip missing sota inference dir:", model_dir)
        return
    return _orig_inf_load(self, model_dir, device_idx)
Diffusion_Predictor.load_model = _safe_inf_load

INDEX_MAPS = {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]}

MODEL_FEATURES = {
    "EfficientMPLight": ["cur_phase", "traffic_movement_pressure_num"],
    "EfficientColight": ["cur_phase", "lane_num_vehicle", "adjacency_matrix"],
}

MODEL_REWARD = {
    "EfficientMPLight": {"pressure": -0.25},
    "EfficientColight": {"queue_length": -0.25},
}

MODEL_NOISE = {
    "EfficientMPLight": {"NOISE_SCALE": 3.5, "NOISE_TYPE": 3},
    "EfficientColight": {"NOISE_SCALE": 4.0, "NOISE_TYPE": 3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="EfficientMPLight",
                    choices=list(MODEL_FEATURES))
    ap.add_argument("--flow",
                    default="anon_4_4_hangzhou_real_5816.json")
    ap.add_argument("--memo", default=None)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--roadnet", default="4_4")
    ap.add_argument("--template", default="Hangzhou")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    memo = args.memo or "ctl_%s_%s" % (args.mod.lower(), args.flow[:-5])
    ncol = int(args.roadnet.split("_")[1])
    nrow = int(args.roadnet.split("_")[0])

    env_extra = {
        "NUM_ROUNDS": args.rounds,
        "NUM_GENERATORS": 1,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": nrow * ncol,
        "RUN_COUNTS": 3600,
        "MODEL_NAME": args.mod,
        "NUM_ROW": nrow,
        "NUM_COL": ncol,
        "TRAFFIC_FILE": args.flow,
        "ROADNET_FILE": "roadnet_%s.json" % args.roadnet,
        "TRAFFIC_SEPARATE": args.flow,
        "LIST_STATE_FEATURE": MODEL_FEATURES[args.mod],
        "DIC_REWARD_INFO": MODEL_REWARD[args.mod],
        "is_test": False,
        "log_writer": False,
        "inference_epoch": 0,
        "is_inference": False,
        "sota_path": "model/%s/%s" % (args.mod.lower(), args.flow[:-5]),
        "NOISE_LEVEL": 0,
        "device": "cpu:0",  # torch-parsable; last char must be int (agent code does int(device[-1]))
        "index_maps": INDEX_MAPS,
        "NOISE_DIRECTION": ["W", "E"],
        "inference_config": Robust_config,
    }
    env_extra.update(MODEL_NOISE[args.mod])

    ts = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    path_extra = {
        "PATH_TO_MODEL": os.path.join("model", memo, args.flow + "_" + ts),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, args.flow + "_" + ts),
        "PATH_TO_DATA": os.path.join("data", args.template, args.roadnet),
        "PATH_TO_ERROR": os.path.join("errors", memo),
    }

    conf = merge(config.dic_traffic_env_conf, env_extra)
    path = merge(config.DIC_PATH, path_extra)
    agent_conf = getattr(config, "DIC_BASE_AGENT_CONF")

    print("mod=%s flow=%s rounds=%d memo=%s" % (args.mod, args.flow, args.rounds, memo))
    print("model dir:", path["PATH_TO_MODEL"])
    pipeline_wrapper(dic_agent_conf=agent_conf,
                     dic_traffic_env_conf=conf,
                     dic_path=path)


if __name__ == "__main__":
    main()
