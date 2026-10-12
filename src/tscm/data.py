"""Trace collection and training dataset.

Collection: closed-loop runs under a chosen controller (default: complete-obs
SMP) storing per-decision q_true and phase indices. Missingness is applied
*offline* during training (artificial masks, LMP-GNN Section 4.4 protocol):
for each sampled decision we re-derive the degraded (Q,h,s) history from
q_true and a freshly sampled mask sequence -> realistic staleness structure.

Sample = (episode, decision d): window [d-T+1 .. d] inputs + target q_true[d]
+ eval mask z_d (which lanes are hidden at the decision step).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .runner import run_closed
from .obs import ObsTracker, lane2tls_map
from .smp import build_pressure_matrices
from .model import build_movement_graph, lane_static_features


def collect_trace(net, rou, horizon=3600, interval=10, seed=0,
                  filler_spec="full", mask=None, window=12):
    """Run closed loop, return per-decision arrays."""
    m, recs = run_closed(net, rou, filler_spec=filler_spec, mask=mask,
                         horizon=horizon, interval=interval, seed=seed,
                         window=window, trace=True)
    D = len(recs)
    if D == 0:
        raise RuntimeError("empty trace")
    L = recs[0]["q"].shape[0]
    tls_ids = sorted({k for r in recs for k in r["s_true"]})
    q_true = np.stack([r["q"] for r in recs])          # [D,L]
    z_sched = np.stack([r["z"] for r in recs])
    phase_idx = np.zeros((D, len(tls_ids)), np.int64)
    for d, r in enumerate(recs):
        for j, t in enumerate(tls_ids):
            phase_idx[d, j] = r["virt"][t]      # green phase in effect at decision d
    return {"q_true": q_true, "z_sched": z_sched, "phase_idx": phase_idx,
            "tls_ids": tls_ids, "metrics": m, "recs": recs}


def world_meta(net, rou, seed=0):
    """Static meta shared by training/inference: graphs, W, static feats, groups."""
    from .world import SumoWorld
    from .obs import lane_groups_for
    w = SumoWorld(net, rou).start()
    lane_index = w.lane_index
    lane_pos = w._lane_pos
    Ws, legal = build_pressure_matrices(w, lane_pos)
    g = build_movement_graph(w, lane_pos)
    stat = lane_static_features(w._net, lane_index)
    l2t = lane2tls_map(w, lane_pos)
    groups = lane_groups_for(w)
    meta = {"lane_index": lane_index, "Ws": Ws, "legal": legal,
            "graph": g, "static": stat, "lane2tls": l2t, "groups": groups,
            "n_phases_max": max(it.n_phases for it in w.intersections),
            "tls_ids": [it.id for it in w.intersections]}
    w.close()
    return meta


# ---------------- mask sequence sampler for training ----------------

def sample_mask_seq(rng, L, T, kind="random", pmax=0.5, groups=None):
    """Sample a mask history [T,L] with a random regime mix."""
    if kind == "random":
        p = rng.uniform(0.05, pmax)
        return (rng.random((T, L)) < p).astype(np.int8)
    if kind == "burst":
        z = np.zeros((T, L), np.int8)
        p = rng.uniform(0.05, pmax)
        n = max(1, int(p * L))
        lanes = rng.choice(L, n, replace=False)
        t0 = rng.integers(0, max(1, T // 2))
        dur = rng.integers(1, T - t0 + 1)
        z[t0:t0 + dur, lanes] = 1
        return z
    if kind == "correlated" and groups:
        z = np.zeros((T, L), np.int8)
        for g in groups:
            if rng.random() < rng.uniform(0.05, pmax):
                t0 = rng.integers(0, T)
                dur = rng.integers(1, T - t0 + 1)
                z[t0:t0 + dur, np.asarray(g)] = 1
        return z
    return (rng.random((T, L)) < rng.uniform(0.05, pmax)).astype(np.int8)


class TraceDataset(Dataset):
    """Windowed decision samples from collected episodes.

    __getitem__ -> dict of torch tensors ready for model.forward + losses.
    """

    KINDS = ("random", "burst", "correlated")

    def __init__(self, episodes, meta, window=12, pmax=0.5, seed=0,
                 kinds=("random", "burst", "correlated")):
        self.eps, self.meta, self.T = episodes, meta, window
        self.pmax, self.rng = pmax, np.random.default_rng(seed)
        self.kinds = kinds
        self.index = [(e, d) for e, ep in enumerate(episodes)
                      for d in range(len(ep["q_true"]))]
        self.static = torch.tensor(meta["static"], dtype=torch.float32)
        self.l2t = meta["lane2tls"]
        self.n_ph = meta["n_phases_max"]

    def __len__(self):
        return len(self.index)

    def _derive_window(self, q_win, z_win):
        """From true counts and mask history -> Q,h,s windows [T,L]."""
        T, L = q_win.shape
        Q = np.where(z_win.astype(bool), 0.0, q_win).astype(np.float32)
        h = np.zeros((T, L), np.float32)
        s = np.zeros((T, L), np.float32)
        last = np.zeros(L, np.float32)
        stale = np.zeros(L, np.float32)
        for t in range(T):
            h[t] = last; s[t] = stale
            obs = z_win[t] == 0
            last = np.where(obs, q_win[t], last)
            stale = np.where(obs, 0.0, stale + 1.0)
        return Q, h, s

    def __getitem__(self, i):
        e, d = self.index[i]
        ep = self.eps[e]
        q_all, L = ep["q_true"], ep["q_true"].shape[1]
        t0 = max(0, d - self.T + 1)
        q_win = q_all[t0:d + 1]
        Tw = q_win.shape[0]
        # pad window on the left by repeating first row (observed)
        if Tw < self.T:
            q_win = np.concatenate([np.repeat(q_win[:1], self.T - Tw, 0), q_win], 0)
        kind = self.kinds[self.rng.integers(len(self.kinds))]
        z_win = sample_mask_seq(self.rng, L, self.T, kind, self.pmax,
                                self.meta["groups"])
        # during left padding, force observed (the state actually was observed)
        z_win[:self.T - Tw] = 0
        Q, h, s = self._derive_window(q_win, z_win)
        vis = np.zeros((self.T, L), np.float32)
        # visible lane mean at the lane's own intersection (LMP-GNN Eq.5 term)
        for t in range(self.T):
            obs = z_win[t] == 0
            vis[t] = np.where(obs, Q[t], np.nan)
            vm = np.zeros(L, np.float32)
            for j in np.unique(self.l2t):
                sel = (self.l2t == j) & obs
                vm[self.l2t == j] = Q[t][sel].mean() if sel.any() else 0.0
            vis[t] = vm
        dyn = np.stack([np.log1p(Q), z_win.astype(np.float32),
                        np.log1p(h), np.minimum(s, 50) / 50.0,
                        np.log1p(np.nan_to_num(vis))], -1)          # [T,L,5]
        # phase context: one-hot of upstream TLS current phase + time-in-phase
        J = ep["phase_idx"].shape[1]
        ph_ctx = np.zeros((self.T, L, self.n_ph + 1), np.float32)
        for t in range(self.T):
            dt = t0 + t - (self.T - Tw)
            dt = min(max(dt, 0), d)
            for j in range(J):
                ph_idx = ep["phase_idx"][dt, j]
                sel = self.l2t == j
                ph_ctx[t, sel, ph_idx] = 1.0
                # normalize position of phase in program
                ph_ctx[t, sel, -1] = ph_idx / max(self.n_ph - 1, 1)
        return {
            "dyn": torch.tensor(dyn),                       # [T,L,5]
            "ctx": torch.tensor(ph_ctx),                    # [T,L,nph+1]
            "h_anchor": torch.tensor(h[-1], dtype=torch.float32),
            "z_eval": torch.tensor(z_win[-1], dtype=torch.float32),
            "y": torch.tensor(q_all[d], dtype=torch.float32),
            "Q_last": torch.tensor(Q[-1], dtype=torch.float32),
            "ep": e, "d": d,
        }
