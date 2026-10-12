"""Closed-loop inference adapter (M1) + phase-level risk decision rule (M3).

ModelAdapter: turns an ObsTracker window (Q,z,h,s,ph [T,L]) into model inputs
and returns (mu, sigma) per lane. Caches last (mu, sigma) so the risk rule
does not pay for a second forward pass in the same decision.

Risk selection (M3, this project's phase-level rule — distinct from
LMP-GNN's per-lane input discounts):
    Var[P(a)] = sum_l W2[a,l] * sigma_l^2      (W2 = W o W; independence)
    s'(a)     = s(a) - kappa * sqrt(Var[P(a)])  (conservative lower bound)
    a_risk    = argmax over legal s'
Optionally 'hold': if top1-top2 score margin < gate * sigma_margin, keep the
current virtual phase (hysteresis against jittery switching).
"""
import numpy as np
import torch


class ModelAdapter:
    """callable(window_stack)->(mu,sigma); keeps last prediction cached."""

    def __init__(self, model, meta, device="cpu"):
        self.model = model.eval()
        self.meta, self.dev = meta, device
        self.static = torch.tensor(meta["static"], dtype=torch.float32, device=device)
        self.g = {k: v.to(device) for k, v in meta["graph"].items()
                  if torch.is_tensor(v)}
        self.l2t = np.asarray(meta["lane2tls"])
        self.n_ph = int(meta["n_phases_max"])
        self.last = None

    @torch.no_grad()
    def __call__(self, w):
        Q, z, h, s = w["Q"], w["z"], w["h"], w["s"]
        T, L = Q.shape
        ph = w["ph"] if "ph" in w else np.zeros((T, L), np.int8)
        vis = np.zeros((T, L), np.float32)
        for t in range(T):
            obs = z[t] == 0
            for j in np.unique(self.l2t):
                sel = (self.l2t == j) & obs
                if sel.any():
                    vis[t][self.l2t == j] = Q[t][sel].mean()
        dyn = np.stack([np.log1p(Q), z.astype(np.float32), np.log1p(h),
                        np.minimum(s, 50) / 50.0, np.log1p(vis)], -1)
        ctx = np.zeros((T, L, self.n_ph + 1), np.float32)
        for t in range(T):
            for k in range(L):
                p = int(ph[t, k])
                if 0 <= p < self.n_ph:
                    ctx[t, k, p] = 1.0
                ctx[t, k, -1] = p / max(self.n_ph - 1, 1)
        dyn_t = torch.tensor(dyn[None], device=self.dev)
        ctx_t = torch.tensor(ctx[None], device=self.dev)
        ha = torch.tensor(h[-1][None], dtype=torch.float32, device=self.dev)
        mu, sigma = self.model(dyn_t, self.static, ctx_t, ha,
                               self.g["MS"], self.g["MD"], self.g["LS"], self.g["LD"])
        self.last = (mu[0].cpu().numpy(), sigma[0].cpu().numpy())
        return self.last


def make_risk_fn(kappa=1.0, gate=0.0, mode="lcb"):
    """M3 decision rule. mode: 'lcb' (score lower bound) or 'lcb_hold'.

    signature: fn(smp, x, snap, tracker, model) -> {tls: phase}
    """
    def decide(smp, x, snap, tracker, model):
        sigma = np.zeros_like(x) if model is None else np.asarray(model.last[1])
        acts = {}
        for it in smp.world.intersections:
            if it.current_phase_time < smp.t_min and it.current_phase_time > 0:
                acts[it.id] = it.virtual_phase
                continue
            W = smp.W[it.id]
            s = np.asarray(W @ x).ravel()
            var = np.asarray((W.multiply(W)) @ (sigma ** 2)).ravel()
            sp_std = np.sqrt(np.maximum(var, 0.0))
            s_lcb = s - kappa * sp_std
            legal = it.legal_phases
            order = sorted(legal, key=lambda p: -s_lcb[p])
            a_risk = order[0]
            if mode == "lcb_hold" and gate > 0 and len(order) > 1:
                margin = s[order[0]] - s[order[1]]
                unc = gate * max(sp_std[order[0]], sp_std[order[1]])
                if margin < unc:
                    a_risk = it.virtual_phase if it.virtual_phase in legal else order[0]
            acts[it.id] = a_risk
        return acts
    return decide
