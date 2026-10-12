"""Missing-value input rules: baselines + LMP-GNN rules + model hook.

Every rule maps decision-time information (Q, z, h, s, model outputs) to a
completed lane vector x, preserving observed entries:
    x_i = (1 - z_i) * q_i + z_i * g_i          (LMP-GNN Eq.4)

Baseline rules (no model):
  zero       -> 0
  last       -> h (last observed count)
  mean       -> expanding mean of that lane's past observations
  road_mean  -> mean of currently-observed lanes in the same approach group
                (LMP-GNN's "Road Mean" spatial baseline)
LMP-GNN rules (Eq.13-15) applied on model outputs (mu, sigma, staleness s):
  pred_mean   -> mu
  fixed_disc  -> max(0, mu - lam * sigma)        lambda = 1
  staleness   -> max(0, mu - lam * exp(-s/tau) * sigma)   tau = 2 decisions
Model hook: `model` is any callable(window_dict) -> (mu, sigma) per lane.
"""
import numpy as np


class Filler:
    name = "base"

    def fill(self, snap, tracker, model=None, groups=None):
        """snap: {Q,z,h,s,q_true} -> completed x. Observed entries pass through."""
        z = snap["z"].astype(bool)
        x = np.where(z, self._fill_missing(snap, tracker, model, groups), snap["Q"])
        return x.astype(np.float32)

    def _fill_missing(self, snap, tracker, model, groups):
        raise NotImplementedError


class ZeroFiller(Filler):
    name = "zero"

    def _fill_missing(self, snap, tracker, model, groups):
        return np.zeros_like(snap["Q"])


class LastObsFiller(Filler):
    name = "last"

    def _fill_missing(self, snap, tracker, model, groups):
        return snap["h"]


class MeanFiller(Filler):
    """Expanding per-lane temporal mean over observations received so far."""
    name = "mean"

    def __init__(self):
        self._sum = None
        self._n = None

    def _fill_missing(self, snap, tracker, model, groups):
        q_obs, z = snap["q_true"], snap["z"].astype(bool)
        if self._sum is None:
            self._sum = np.zeros_like(q_obs, np.float64)
            self._n = np.zeros_like(q_obs, np.float64)
        obs = ~z
        self._sum[obs] += q_obs[obs]
        self._n[obs] += 1
        m = self._n > 0
        # lanes never observed fall back to h (0) — same as last-obs baseline
        return np.where(m, self._sum / np.maximum(self._n, 1), snap["h"]).astype(np.float32)


class RoadMeanFiller(Filler):
    """Spatial mean over observed lanes in the same approach group."""
    name = "road_mean"

    def _fill_missing(self, snap, tracker, model, groups):
        z = snap["z"].astype(bool)
        Q, out = snap["Q"], np.zeros_like(snap["Q"])
        covered = np.zeros_like(z)
        for g in (groups or []):
            g = np.asarray(g)
            miss = z[g]
            if not miss.any():
                continue
            obsv = Q[g][~miss]
            if len(obsv):
                out[g[miss]] = obsv.mean()
                covered[g[miss]] = True
        # fully-missing groups / ungrouped lanes: temporal fallback to h
        uc = z & ~covered
        out[uc] = snap["h"][uc]
        return out


class ModelFiller(Filler):
    """Wraps a recovery model + an LMP-GNN input rule (Eq.13-15)."""

    def __init__(self, rule="pred_mean", lam=1.0, tau=2.0):
        self.rule, self.lam, self.tau = rule, lam, tau
        self.name = f"model_{rule}"

    def _fill_missing(self, snap, tracker, model, groups):
        mu, sigma = model(tracker.window_stack())
        mu = np.asarray(mu, np.float32)
        sigma = np.asarray(sigma, np.float32)
        s = snap["s"]
        if self.rule == "pred_mean":
            g = mu
        elif self.rule == "fixed_disc":
            g = np.maximum(0.0, mu - self.lam * sigma)
        elif self.rule == "staleness":
            g = np.maximum(0.0, mu - self.lam * np.exp(-s / self.tau) * sigma)
        else:
            raise ValueError(self.rule)
        return g


def make_filler(spec):
    """'zero'|'last'|'mean'|'road_mean'|'model:<rule>[:lam=x:tau=y]'|'full'"""
    if spec in ("full", "oracle"):
        return None
    table = {"zero": ZeroFiller, "last": LastObsFiller,
             "mean": MeanFiller, "road_mean": RoadMeanFiller}
    if spec in table:
        return table[spec]()
    if spec.startswith("model"):
        parts = spec.split(":")
        rule = parts[1] if len(parts) > 1 else "pred_mean"
        kw = dict(kv.split("=") for kv in parts[2:] if "=" in kv)
        return ModelFiller(rule, float(kw.get("lam", 1.0)), float(kw.get("tau", 2.0)))
    raise ValueError(spec)
