#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate demand variants of a CityFlow flow JSON for ImputeFormer training data.

Each variant: subsample vehicles with keep_prob ~ U[k_lo, k_hi] and jitter
startTime/endTime by +-JIT seconds. Routes are kept intact.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-lo", type=float, default=0.70)
    ap.add_argument("--keep-hi", type=float, default=1.00)
    ap.add_argument("--jit", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    flows = json.load(open(args.src))
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.src))[0]

    for k in range(args.n):
        keep_p = rng.uniform(args.keep_lo, args.keep_hi)
        out = []
        for veh in flows:
            if rng.rand() > keep_p:
                continue
            v = dict(veh)
            s = int(v.get("startTime", 0)) + rng.randint(-args.jit, args.jit + 1)
            e = int(v.get("endTime", 0)) + rng.randint(-args.jit, args.jit + 1)
            s = max(0, s)
            e = max(s, e)
            v["startTime"] = s
            v["endTime"] = e
            out.append(v)
        name = "{0}_var{1:03d}.json".format(base, k)
        with open(os.path.join(args.outdir, name), "w") as f:
            json.dump(out, f)
        print("variant {0}: keep_p={1:.3f} n_veh={2}/{3}".format(name, keep_p, len(out), len(flows)))


if __name__ == "__main__":
    main()
