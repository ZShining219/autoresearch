"""Closed-loop experiment driver and stored-trace decision audit.

Closed loop (per decision, every `interval` seconds):
    q = true lane counts -> mask z -> completed x -> SMP/risk decision -> advance
Decision-time diagnostics (recorded every decision):
    scores under complete q vs under filled x on the SAME state
    -> phase agreement, pressure-score error, pressure regret.
Stored-trace audit replays saved states offline for identical-state comparison.
"""
import time
import numpy as np

from .world import SumoWorld
from .smp import SignedMaxPressure, phase_scores, argmax_first
from .obs import MaskScheduler, ObsTracker, lane_groups_for, lane2tls_map
from .rules import make_filler


def run_closed(net, rou, filler_spec="full", mask=None, model=None,
               decision_fn=None, t_min=10, interval=10, horizon=3600,
               yellow=3, seed=0, window=12, trace=False, verbose=False):
    """One closed-loop episode. Returns metrics + (optional) per-decision trace.

    filler_spec: 'full'|'zero'|'last'|'mean'|'road_mean'|'model:<rule>'
    model: callable(window_stack)->(mu,sigma) for ModelFiller
    decision_fn: optional callable(smp, x, extra)->actions overriding SMP choice (M3)
    """
    world = SumoWorld(net, rou, yellow_len=yellow, seed=seed).start()
    smp = SignedMaxPressure(world, world._lane_pos, t_min=t_min)
    n = len(world.lane_index)
    tracker = ObsTracker(n, window)
    groups = lane_groups_for(world)
    filler = make_filler(filler_spec)
    l2t = lane2tls_map(world, world._lane_pos)
    n_dec = int(horizon // interval) + 2
    recs = []
    infer_ms = []

    d = 0
    while world.time < horizon:
        q = world.lane_counts()
        z = mask.at(d) if mask is not None else np.zeros(n, np.int8)
        virt = {it.id: it.virtual_phase for it in world.intersections}
        virt_idx = np.array([it.virtual_phase for it in world.intersections])
        ph_lane = virt_idx[l2t].astype(np.int8)        # phase in effect per lane
        snap = tracker.update(q, z, extras={"ph": ph_lane})
        t0 = time.perf_counter()
        x = q if filler is None else filler.fill(snap, tracker, model, groups)
        if decision_fn is None:
            acts = smp.decide(x)
        else:
            acts = decision_fn(smp, x, snap, tracker, model)
        infer_ms.append((time.perf_counter() - t0) * 1e3)
        if trace:
            s_true = smp.scores(q)
            s_fill = smp.scores(x)
            recs.append({"t": world.time, "q": q.copy(), "z": z.copy(),
                         "x": x.copy(), "s_true": s_true, "s_fill": s_fill,
                         "act": dict(acts), "virt": dict(virt)})
        world.step_seconds(acts, int(interval))
        d += 1

    m = world.metrics()
    m["decisions"] = d
    m["infer_ms_median"] = float(np.median(infer_ms)) if infer_ms else 0.0
    m["infer_ms_p95"] = float(np.percentile(infer_ms, 95)) if infer_ms else 0.0
    if verbose:
        print(f"[{filler_spec}] ATT={m['avg_travel_time']:.1f} tp={m['throughput']} "
              f"unf={m['unfinished']} q~{m['avg_queue']:.0f}/{m['max_queue']:.0f} "
              f"infer={m['infer_ms_median']:.2f}ms")
    world.close()
    return m, recs


# ---------------- decision-time audit on a trace ----------------

def audit_trace(recs):
    """Per-decision diagnostics along one closed-loop trajectory.

    Returns arrays of: masked-lane count error, pressure-score error,
    phase disagreement (top1 mismatch vs complete-obs argmax), regret
    (true-score gap between the two argmax choices).
    """
    cnt_err, scr_err, disagree, regret, margin = [], [], [], [], []
    for r in recs:
        z = r["z"].astype(bool)
        err = np.abs(r["x"] - r["q"])[z]
        cnt_err.append(err.mean() if err.size else 0.0)
        se, dg, rg, mg = [], [], [], []
        for tls, s_t in r["s_true"].items():
            s_f = r["s_fill"][tls]
            legal = np.isfinite(s_t)
            st, sf = s_t[legal], s_f[legal]
            se.append(np.abs(st - sf).mean())
            a_t = legal.nonzero()[0][np.argmax(st)]
            a_f = legal.nonzero()[0][np.argmax(sf)]
            dg.append(float(a_t != a_f))
            rg.append(s_t[a_t] - s_t[a_f])
            srt = np.sort(st)[::-1]
            mg.append(srt[0] - srt[1] if len(srt) > 1 else np.inf)
        scr_err.append(np.mean(se)); disagree.append(np.mean(dg))
        regret.append(np.mean(rg)); margin.append(np.mean(mg))
    return {k: np.asarray(v, np.float64) for k, v in
            dict(count_mae=cnt_err, score_mae=scr_err, disagree=disagree,
                 regret=regret, top_margin=margin).items()}
