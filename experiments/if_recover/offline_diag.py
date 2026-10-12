#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline diagnosis of IF recovery vs zero-fill on direction-masked windows.

For each held-out window with a persistent direction mask:
  - IF imputation of missing positions
  - zero-fill baseline (what the masked arm effectively feeds)
Reports MAE and signed bias (y_hat - y_true) on missing positions, per direction.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = "/data/users/zfh/workspace/projects/auto_research"
sys.path.insert(0, os.path.join(ROOT, "experiments", "if_recover"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import if_data  # noqa: E402
from ifrec.recoverer import IFRecoverer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "results/if_recover/model_if.pt"))
    ap.add_argument("--traces", default=os.path.join(ROOT, "data/if_recover/traces"))
    ap.add_argument("--pat", default="orig_5816.npz")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    rec = IFRecoverer(args.ckpt)
    rng = np.random.RandomState(0)
    X, U = if_data.build_window_bank(args.traces, args.window, [args.pat])
    X, U = X[:args.n], U[:args.n]
    print("windows:", X.shape)

    dev = "cpu"
    model = rec.model
    for dire in ["W", "W,E", "N", "S"]:
        dirs = dire.split(",")
        nodes = if_data._dir_nodes(dirs)
        errs_if, bias_if, errs0 = [], [], []
        for i in range(len(X)):
            xw, uw = X[i:i + 1], U[i:i + 1]          # [1,w,192,1],[1,w,192,4]
            m = if_data.mask_dir_global(rng, args.window, directions=dirs)  # [w,192] True=miss
            xin = xw.copy()
            xin[0, m, 0] = 0.0                        # zero-fill the missing
            obs = torch.tensor((~m)[None, :, :, None].astype(np.float32))
            xt = torch.tensor((xin - rec.mu) / rec.sd)
            ut = torch.tensor(uw)
            with torch.no_grad():
                yh = model(xt, ut, obs) * rec.sd + rec.mu
            y_true = xw[0, m, 0]
            y_if = yh[0, :, :, 0].cpu().numpy()[m]
            errs_if.append(np.abs(y_if - y_true))
            bias_if.append(y_if - y_true)
            errs0.append(np.abs(0.0 - y_true))
        print(f"{dire:4s} n={int(np.concatenate([e.ravel() for e in errs_if]).size):6d} "
              f"IF MAE={np.mean(np.concatenate(errs_if)):.3f} "
              f"zero MAE={np.mean(np.concatenate(errs0)):.3f} "
              f"IF bias(mean-signed)={np.mean(np.concatenate(bias_if)):+.3f} "
              f"|true|mean={np.mean(np.abs(np.concatenate(errs0))):.3f}")


if __name__ == "__main__":
    main()
