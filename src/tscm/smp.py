"""Signed Max-Pressure (SMP) controller and pressure algebra.

Pressure definition (LMP-GNN Eq.1-2; identical to LibSignal maxpressure):
    P_{j,p}(x) = sum_{(u,d) in E_{j,p}} (x_u - x_d)
    a*         = argmax_{p in Phi_j, E_{j,p} != empty} P_{j,p}(x)
Ties: first maximum in phase order (CityFlow/LibSignal convention).

The controller operates on a *completed* lane-count vector x (observed
counts passthrough + filled missing counts). The same matrix W is reused
for score computation and for differentiable decision losses (M2) and
uncertainty propagation (M3): P(x) = W @ x with W in {+1,-1,0}.
"""
import numpy as np
import scipy.sparse as sp


def build_pressure_matrices(world, lane_pos):
    """Return per-intersection CSR matrix W_j of shape [n_phases, n_lanes].

    W_j[p, l] = (#lanelinks where l is upstream in phase p)
              - (#lanelinks where l is downstream in phase p)
    Only legal phases (nonempty lanelink set) are kept, per Eq.2.
    """
    n = len(lane_pos)
    Ws = {}
    legal = {}
    for it in world.intersections:
        rows, cols, data = [], [], []
        for p in it.legal_phases:
            for fr, to in it.phase_available_lanelinks[p]:
                rows += [p, p]; cols += [lane_pos[fr], lane_pos[to]]; data += [1.0, -1.0]
        W = sp.csr_matrix((data, (rows, cols)), shape=(it.n_phases, n))
        Ws[it.id] = W
        legal[it.id] = it.legal_phases
    return Ws, legal


def phase_scores(W, x, legal_phases=None):
    """Score vector over the full phase index set; -inf for illegal phases."""
    s = np.asarray(W @ x).ravel()
    if legal_phases is not None:
        out = np.full_like(s, -np.inf, dtype=np.float64)
        out[legal_phases] = s[legal_phases]
        return out
    return s


def argmax_first(scores, legal_phases):
    """argmax over legal phases, first-max tie-break."""
    best, bs = None, -np.inf
    for p in legal_phases:
        if scores[p] > bs:
            bs = scores[p]
            best = p
    return best, bs


class SignedMaxPressure:
    """SMP controller: pick argmax pressure phase per intersection, t_min hold."""

    def __init__(self, world, lane_pos, t_min=10):
        self.W, self.legal = build_pressure_matrices(world, lane_pos)
        self.t_min = t_min
        self.world = world

    def decide(self, x):
        """x: completed lane counts -> dict tls_id -> phase index."""
        acts = {}
        for it in self.world.intersections:
            if it.current_phase_time < self.t_min and it.current_phase_time > 0:
                acts[it.id] = it.virtual_phase
                continue
            s = phase_scores(self.W[it.id], x, it.legal_phases)
            acts[it.id], _ = argmax_first(s, it.legal_phases)
        return acts

    def scores(self, x):
        """Full per-intersection score dict for audit/risk use."""
        return {it.id: phase_scores(self.W[it.id], x, it.legal_phases)
                for it in self.world.intersections}
