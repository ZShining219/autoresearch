#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe: same masked closed-loop, compare true vs v1 vs v2 on missing dims.

Controller consumes the MASKED state (0-fill) -> trajectory = masked baseline.
Both recoverers observe identical masked inputs each step; v1 = windowed
inpainting (zeros stay zero in history), v2 = autoreg (own predictions fed
back as observed). Dumps per-step values at missing dims for inspection.
"""
import argparse
import os
import shutil
import sys

import numpy as np

ROOT = "/data/users/zfh/workspace/projects/auto_research"
sys.path.insert(0, os.path.join(ROOT, "experiments", "if_recover"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from eval_closed import CONF, DATA_DIR, MPController, convert, make_mask  # noqa: E402
from utils.cityflow_env import CityFlowEnv  # noqa: E402
from ifrec.recoverer import IFRecoverer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default=os.path.join(
        DATA_DIR, "anon_4_4_hangzhou_real.json"))
    ap.add_argument("--dirs", default="W")
    ap.add_argument("--ckpt-v1", default=os.path.join(
        ROOT, "results/if_recover/model_if_baseline_ep34.pt"))
    ap.add_argument("--ckpt-v2", default=os.path.join(
        ROOT, "results/if_recover/model_if_sc.pt"))
    ap.add_argument("--workdir", default=os.path.join(
        ROOT, "data/if_recover/rundirs/probe"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "data/if_recover/probe_values.npz"))
    ap.add_argument("--inter", type=int, default=0)
    args = ap.parse_args()

    conf = dict(CONF)
    conf["TRAFFIC_FILE"] = os.path.basename(args.flow)
    if os.path.exists(args.workdir):
        shutil.rmtree(args.workdir)
    log_dir = os.path.join(args.workdir, "test_round", "round_0")
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(args.flow, os.path.join(args.workdir, os.path.basename(args.flow)))
    shutil.copy(os.path.join(DATA_DIR, "roadnet_4_4.json"),
                os.path.join(args.workdir, "roadnet_4_4.json"))
    env = CityFlowEnv(path_to_log=log_dir, path_to_work_directory=args.workdir,
                      dic_traffic_env_conf=conf)

    mps = [MPController() for _ in range(16)]
    v1 = IFRecoverer(args.ckpt_v1, autoreg=False)
    v2 = IFRecoverer(args.ckpt_v2, autoreg=True)
    dirs = [d for d in args.dirs.split(",") if d]
    mask = make_mask(dirs)
    miss = mask < 0.5

    state = env.reset()
    max_steps = int(conf["RUN_COUNTS"] / conf["MIN_ACTION_TIME"])
    rec = {"true": [], "masked": [], "v1": [], "v2": [], "action": []}
    step_num, done = 0, False

    while not done and step_num < max_steps:
        action_list = [mps[i].choose(state[i]) for i in range(16)]
        next_state, reward, done, _ = env.step(action_list)
        xs = convert(next_state)
        x_masked = mask * xs
        aoh = np.eye(4, dtype=np.float32)[np.asarray(action_list)]
        o1 = v1.demask_state(x_masked, aoh, None, mask)
        o2 = v2.demask_state(x_masked, aoh, None, mask)
        rec["true"].append(xs[miss])
        rec["masked"].append(x_masked[miss])
        rec["v1"].append(o1[miss])
        rec["v2"].append(o2[miss])
        rec["action"].append(list(action_list))
        # controller consumes masked (0-fill) state -> masked baseline trajectory
        if step_num >= 4:
            for i, s in enumerate(next_state):
                s["traffic_movement_pressure_queue"] = x_masked[i].tolist()
        state = next_state
        step_num += 1

    arr = {k: np.stack(v) for k, v in rec.items()}
    np.savez_compressed(args.out, **arr, miss_dims=np.argwhere(miss[0]).ravel())
    print("saved:", args.out, "steps:", step_num, "missing dims/inter:", miss[0].sum())

    # readable slice: chosen intersection's missing dims
    md = np.argwhere(miss[0]).ravel()          # missing dim indices per inter
    cnt_per_inter = int(miss[0].sum())
    lo, hi = args.inter * cnt_per_inter, (args.inter + 1) * cnt_per_inter
    T = arr["true"].shape[0]
    show = list(range(0, 60, 2)) + list(range(60, T, 30))
    print(f"\ninter={args.inter} missing dims={md.tolist()} (W: WL,WT,WR slots)")
    hdr = "step | " + " | ".join(f"dim{d}:true,mask,v1,v2" for d in md)
    print(hdr)
    for t in show:
        if t >= T:
            break
        tr = arr["true"][t, lo:hi]
        mk = arr["masked"][t, lo:hi]
        a = arr["v1"][t, lo:hi]
        b = arr["v2"][t, lo:hi]
        row = " | ".join(f"{tr[j]:7.2f},{mk[j]:5.1f},{a[j]:7.2f},{b[j]:7.2f}"
                         for j in range(len(md)))
        print(f"{t:4d} | {row}")
    print("\nMAE summary  v1: %.3f  v2: %.3f  (on all missing dims, all steps)" % (
        np.abs(arr["v1"] - arr["true"]).mean(),
        np.abs(arr["v2"] - arr["true"]).mean()))
    print("true mean on missing dims: %.3f" % arr["true"].mean())


if __name__ == "__main__":
    main()
