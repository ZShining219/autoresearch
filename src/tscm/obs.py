"""Observation pipeline: mask schedules + decision-time information state.

Implements LMP-GNN Section 3.3 information boundary:
  q_t    true lane counts (simulator, hidden from controller when masked)
  z_t    missingness mask, 1 = unavailable
  Q_t    stored counts = (1 - z_t) * q_t        (Eq.3)
  h_t    last observed count per lane
  s_t    decisions since last valid observation (staleness)
  F_t    = observations/masks up to t + h_t + s_t + fixed network info

Mask schedules are generated once per (seed, scenario) and shared by all
compared methods (matched outage schedule, LMP-GNN Section 5.1).
"""
import numpy as np


class MaskScheduler:
    """Precomputed per-decision missingness z_t over lanes.

    kinds:
      random     iid Bernoulli(rate) per lane per decision
      correlated approach-grouped: each decision picks random approach groups
                 (all its incoming lanes) with prob rate -> spatially correlated
      sustained  fixed lanes missing for a contiguous window [t0, t0+dur)
      blackout   all lanes of one approach per intersection missing whole run
    """

    def __init__(self, n_lanes, lane_groups, n_decisions, kind, rate, seed,
                 dur_decisions=30, n_groups_frac=1.0):
        self.n_lanes, self.n_decisions = n_lanes, n_decisions
        self.kind, self.rate = kind, rate
        rng = np.random.default_rng(seed)
        self.z = np.zeros((n_decisions, n_lanes), dtype=np.int8)
        if kind == "random":
            self.z[:] = (rng.random(self.z.shape) < rate)
        elif kind == "correlated":
            # per decision, each approach group masked with prob `rate` —
            # lane-level rate = rate (group-level granularity)
            for t in range(n_decisions):
                for g in lane_groups:
                    if rng.random() < rate * n_groups_frac:
                        self.z[t, g] = 1
        elif kind == "sustained":
            t0 = n_decisions // 3
            k = max(1, int(round(rate * n_lanes)))
            lanes = rng.choice(n_lanes, size=k, replace=False)
            self.z[t0:t0 + dur_decisions, lanes] = 1
        elif kind == "blackout":
            for g in lane_groups:
                if rng.random() < rate:
                    self.z[:, g] = 1
        else:
            raise ValueError(kind)

    def at(self, t):
        return self.z[min(t, self.n_decisions - 1)]


class ObsTracker:
    """Maintains Q/z/h/s and a rolling window of decision-time features."""

    def __init__(self, n_lanes, window=12):
        self.n, self.window = n_lanes, window
        self.h = np.zeros(n_lanes, np.float32)      # last observed count
        self.s = np.zeros(n_lanes, np.float32)      # staleness (decisions)
        self.hist = []                               # deque of feature dicts
        self.seen = np.zeros(n_lanes, bool)          # ever observed

    def update(self, q, z, extras=None):
        """Record one decision step; returns decision-time feature dict.

        q: true counts (only used where z==0 for Q/h; full q also kept for eval)
        z: missing mask
        extras: per-lane broadcast exogenous features (e.g., phase one-hot is
                handled by caller via separate arrays)
        """
        z = z.astype(bool)
        Q = np.where(z, 0.0, q).astype(np.float32)
        # update h/s for NEXT step (obs processed after decision)
        snap = {
            "Q": Q.copy(), "z": z.astype(np.int8).copy(),
            "h": self.h.copy(), "s": self.s.copy(),
            "q_true": np.asarray(q, np.float32).copy(),
        }
        if extras:
            snap.update(extras)
        self.hist.append(snap)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        # advance last-obs/staleness: observed lanes refresh, missing stale++
        self.h = np.where(z, self.h, q).astype(np.float32)
        self.seen |= ~z
        self.s = np.where(z, self.s + 1.0, 0.0)
        # never-observed lanes: h stays 0, s grows — treated as missing anyway
        return snap

    def window_stack(self):
        """Return [T, n] arrays of the rolling window (pad by repeating first)."""
        T = self.window
        pad = T - len(self.hist)
        seq = ([self.hist[0]] * pad if pad and self.hist else []) + self.hist
        keys = list(seq[0].keys()) if seq else ["Q", "z", "h", "s", "q_true"]
        return {k: (np.stack([d[k] for d in seq], 0) if seq
                    else np.zeros((T, self.n))) for k in keys}


def lane2tls_map(world, lane_pos):
    """Map each lane to the TLS index whose lanelinks it *feeds* (from-lane)."""
    m = np.full(len(lane_pos), -1, np.int64)
    for j, it in enumerate(world.intersections):
        for ll in it.phase_available_lanelinks:
            for fr, _ in ll:
                if fr in lane_pos:
                    m[lane_pos[fr]] = j
    for j, it in enumerate(world.intersections):
        for ll in it.phase_available_lanelinks:
            for _, to in ll:
                if to in lane_pos and m[lane_pos[to]] < 0:
                    m[lane_pos[to]] = j
    return np.clip(m, 0, None)


def lane_groups_for(world):
    """Group lane indices by incoming edge per intersection (for correlated masks).

    A group = lanes of one incoming edge at one intersection (approach).
    Returns list[list[int]] of lane-index groups.
    """
    groups = []
    for it in world.intersections:
        by_edge = {}
        for ll in it.phase_available_lanelinks:
            for fr, to in ll:
                if fr in world._lane_pos:
                    e = fr.rsplit("_", 1)[0]
                    by_edge.setdefault((it.id, e), set()).add(world._lane_pos[fr])
                if to in world._lane_pos:
                    e = to.rsplit("_", 1)[0]
                    by_edge.setdefault((it.id, e), set()).add(world._lane_pos[to])
        groups.extend(sorted(g) for g in by_edge.values())
    return groups
