#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ImputeFormer data assembly for the RobustLight replacement task.

Node layout: node = intersection_id * 12 + movement_slot  (16 x 12 = 192 nodes)
Value channel: traffic_movement_pressure_queue (the feature RobustLight masks).
Exog channel (u): cur_phase one-hot (4 phases) broadcast to the 12 nodes of its
intersection — always observed, analogous to the diffusion's action conditioning.

Mask semantics: mask[t, n] = True means MISSING (hidden from the model input and
counted in the supervised loss).
"""
import glob
import os

import numpy as np

INDEX_MAPS = {"W": [0, 1, 2], "E": [3, 4, 5], "N": [6, 7, 8], "S": [9, 10, 11]}
N_INTER = 16
N_MOVE = 12
N_NODE = N_INTER * N_MOVE
N_PHASE = 4


def load_episode(npz_path):
    d = np.load(npz_path)
    pq = d["pq"].astype(np.float32)          # [T,16,12]
    phase = d["phase"].astype(np.int64)      # [T,16]
    T = pq.shape[0]
    x = pq.reshape(T, N_NODE, 1)             # [T,192,1]
    return {"x": x, "phase": phase, "att": float(d["att"])}


def phase_onehot(phase):
    """phase [T,16] values in {1..4} -> u [T,192,4] (broadcast to movement nodes)."""
    T = phase.shape[0]
    ph = np.clip(phase - 1, 0, N_PHASE - 1)
    u_inter = np.eye(N_PHASE, dtype=np.float32)[ph]            # [T,16,4]
    u = np.repeat(u_inter, N_MOVE, axis=1)                     # [T,192,4]
    return u


def iter_windows(episode, window):
    """Yield (x_w [w,192,1], u_w [w,192,4]) sliding views, stride 1."""
    x, phase = episode["x"], episode["phase"]
    u = phase_onehot(phase)
    T = x.shape[0]
    for t0 in range(0, T - window + 1):
        yield x[t0:t0 + window], u[t0:t0 + window]


# ---------------- structured mask samplers ----------------
# all return boolean array [window, N_NODE], True = missing

def _dir_nodes(directions, inters=None):
    nodes = []
    if inters is None:
        inters = range(N_INTER)
    for i in inters:
        for dire in directions:
            for m in INDEX_MAPS[dire]:
                nodes.append(i * N_MOVE + m)
    return nodes


def mask_dir_global(rng, window, n_dirs=None, directions=None):
    """Eval-like: chosen directions missing at ALL intersections, whole window."""
    dirs = list(directions) if directions else list(
        rng.choice(list(INDEX_MAPS), size=(n_dirs or rng.choice([1, 2])), replace=False))
    m = np.zeros((window, N_NODE), dtype=bool)
    m[:, _dir_nodes(dirs)] = True
    return m


def mask_dir_subset(rng, window):
    """A direction group missing at a random subset of intersections, whole window."""
    dirs = list(rng.choice(list(INDEX_MAPS), size=rng.choice([1, 2]), replace=False))
    p = rng.uniform(0.2, 1.0)
    inters = [i for i in range(N_INTER) if rng.rand() < p] or [int(rng.randint(N_INTER))]
    m = np.zeros((window, N_NODE), dtype=bool)
    m[:, _dir_nodes(dirs, inters)] = True
    return m


def mask_dir_partial(rng, window):
    """A direction group missing over a random contiguous sub-span of the window."""
    dirs = [rng.choice(list(INDEX_MAPS))]
    m = np.zeros((window, N_NODE), dtype=bool)
    t0 = rng.randint(0, window - 1)
    t1 = min(window, t0 + rng.randint(1, window - t0 + 1))
    m[t0:t1, _dir_nodes(dirs)] = True
    return m


def mask_point(rng, window, p=None):
    return rng.rand(window, N_NODE) < (p if p is not None else rng.uniform(0.05, 0.4))


def mask_block(rng, window, max_len=None):
    """Per-node sustained missing spans (like tsl sample_mask)."""
    m = np.zeros((window, N_NODE), dtype=bool)
    L = max_len or rng.randint(2, window // 2)
    for n in range(N_NODE):
        if rng.rand() < rng.uniform(0.05, 0.5):
            t0 = rng.randint(0, window - 1)
            t1 = min(window, t0 + rng.randint(1, L + 1))
            m[t0:t1, n] = True
    return m


SAMPLERS = {
    "dir_global": mask_dir_global,
    "dir_subset": mask_dir_subset,
    "dir_partial": mask_dir_partial,
    "point": mask_point,
    "block": mask_block,
}
# eval-like persistent direction masks dominate; generic patterns add coverage
MIX_WEIGHTS = {"dir_global": 0.40, "dir_subset": 0.15, "dir_partial": 0.15,
               "point": 0.15, "block": 0.15}


def sample_train_mask(rng, window):
    kind = rng.choice(list(SAMPLERS), p=[MIX_WEIGHTS[k] for k in SAMPLERS])
    return SAMPLERS[kind](rng, window), kind


def build_window_bank(trace_dir, window, patterns):
    """Load episodes and slice windows -> arrays X [B,w,192,1], U [B,w,192,4]."""
    Xs, Us = [], []
    for pat in patterns:
        for f in sorted(glob.glob(os.path.join(trace_dir, pat))):
            ep = load_episode(f)
            for xw, uw in iter_windows(ep, window):
                Xs.append(xw)
                Us.append(uw)
    return np.stack(Xs), np.stack(Us)


def fit_scaler(X):
    return float(X.mean()), float(X.std() + 1e-8)
