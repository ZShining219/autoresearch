"""Early-stop helpers for TSC training.

- plateau: current ATT vs running best (same demand file). Homo / test ATT.
- paired_cycle: file-aligned ATT deltas (legacy).
- cycle_mean: mean of per-episode medians over a demand-set rotation.
  L1/L2 stop on median waiting time: m(t) vs m(t-1) with a relative+absolute dead zone.
"""

from __future__ import annotations

import math


def paired_cycle_decision(prev, curr, min_delta=0.003, z=1.0):
    """Compare two aligned demand-file cycles (legacy ATT pairing)."""
    if len(prev) != len(curr) or not curr:
        raise ValueError("prev and curr must be non-empty and the same length")
    n = len(curr)
    deltas = [float(p) - float(c) for p, c in zip(prev, curr)]
    mean_delta = sum(deltas) / n
    mean_att = sum(float(c) for c in curr) / n
    if n >= 2:
        var = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0
    rel_bar = float(min_delta) * mean_att
    noise_bar = float(z) * se
    threshold = max(rel_bar, noise_bar)
    return {
        "n": n,
        "mean_att": mean_att,
        "mean_delta": mean_delta,
        "se": se,
        "rel_bar": rel_bar,
        "noise_bar": noise_bar,
        "threshold": threshold,
        "improved": mean_delta > threshold,
        "deltas": deltas,
    }


class PairedCycleStopper:
    """Accumulate per-episode values and stop after plateaued paired cycles."""

    def __init__(self, cycle_len, patience, min_delta=0.003, z=1.0, min_episodes=0):
        if cycle_len < 1:
            raise ValueError("cycle_len must be >= 1")
        self.cycle_len = int(cycle_len)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.z = float(z)
        self.min_episodes = int(min_episodes)
        self.atts = []
        self.cycles_without_improve = 0
        self.completed_cycles = 0
        self.last_decision = None

    def add(self, att):
        if att is None or not math.isfinite(float(att)):
            return False, None
        self.atts.append(float(att))
        n = len(self.atts)
        if n < 2 * self.cycle_len or n % self.cycle_len != 0:
            return False, None

        self.completed_cycles = n // self.cycle_len
        curr = self.atts[-self.cycle_len:]
        prev = self.atts[-2 * self.cycle_len:-self.cycle_len]
        decision = paired_cycle_decision(
            prev, curr, min_delta=self.min_delta, z=self.z
        )
        decision["cycle"] = self.completed_cycles - 1
        decision["episodes"] = n
        if decision["improved"]:
            self.cycles_without_improve = 0
        else:
            self.cycles_without_improve += 1
        decision["cycles_without_improve"] = self.cycles_without_improve
        self.last_decision = decision

        stop = (
            n >= self.min_episodes
            and self.patience > 0
            and self.cycles_without_improve >= self.patience
        )
        return stop, decision


def cycle_mean_decision(prev, curr, min_delta=0.05, abs_floor=1.0):
    """Compare rotation means m(t) vs m(t-1). Lower is better.

    Not improving (flat) if |rel| < min_delta OR |delta| < abs_floor.
    Improved only if the mean dropped by more than that dead zone.
    """
    if len(prev) != len(curr) or not curr:
        raise ValueError("prev and curr must be non-empty and the same length")
    m_prev = sum(float(x) for x in prev) / len(prev)
    m_curr = sum(float(x) for x in curr) / len(curr)
    delta = m_curr - m_prev
    if m_prev == 0.0:
        rel = 0.0 if m_curr == 0.0 else float("inf")
    else:
        rel = delta / m_prev
    flat = abs(rel) < float(min_delta) or abs(delta) < float(abs_floor)
    improved = (not flat) and (m_curr < m_prev)
    return {
        "n": len(curr),
        "mean_prev": m_prev,
        "mean_curr": m_curr,
        "delta": delta,
        "rel": rel,
        "min_delta": float(min_delta),
        "abs_floor": float(abs_floor),
        "flat": flat,
        "improved": improved,
    }


class CycleMeanStopper:
    """Stop when the mean of a demand-set rotation of episode medians plateaus."""

    def __init__(self, cycle_len, patience, min_delta=0.05, abs_floor=1.0, min_episodes=0):
        if cycle_len < 1:
            raise ValueError("cycle_len must be >= 1")
        self.cycle_len = int(cycle_len)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.abs_floor = float(abs_floor)
        self.min_episodes = int(min_episodes)
        self.values = []
        self.cycles_without_improve = 0
        self.completed_cycles = 0
        self.last_decision = None

    def add(self, value):
        if value is None or not math.isfinite(float(value)):
            return False, None
        self.values.append(float(value))
        n = len(self.values)
        if n < 2 * self.cycle_len or n % self.cycle_len != 0:
            return False, None

        self.completed_cycles = n // self.cycle_len
        curr = self.values[-self.cycle_len:]
        prev = self.values[-2 * self.cycle_len:-self.cycle_len]
        decision = cycle_mean_decision(
            prev, curr, min_delta=self.min_delta, abs_floor=self.abs_floor
        )
        decision["cycle"] = self.completed_cycles - 1
        decision["episodes"] = n
        if decision["improved"]:
            self.cycles_without_improve = 0
        else:
            self.cycles_without_improve += 1
        decision["cycles_without_improve"] = self.cycles_without_improve
        self.last_decision = decision

        stop = (
            n >= self.min_episodes
            and self.patience > 0
            and self.cycles_without_improve >= self.patience
        )
        return stop, decision
