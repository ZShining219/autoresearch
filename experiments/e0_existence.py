"""E0: problem-existence experiment.

Grid: scenario x missing-kind x rate x fill-rule x seed.
Records closed-loop metrics + along-trajectory decision audit.
Writes results/e0/<tag>.csv and results/e0/audit_<tag>.npz.
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tscm.runner import run_closed, audit_trace
from tscm.scenarios import resolve, SCENARIOS
from tscm.obs import MaskScheduler, lane_groups_for
from tscm.world import SumoWorld


def probe_n_lanes(net, rou):
    w = SumoWorld(net, rou).start()
    n, groups = len(w.lane_index), lane_groups_for(w)
    w.close()
    return n, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="resco4x4")
    ap.add_argument("--kinds", default="random,correlated")
    ap.add_argument("--rates", default="0.1,0.3,0.5")
    ap.add_argument("--fills", default="full,zero,last,mean,road_mean")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "..", "results", "e0")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for seed_s in args.seeds.split(","):
        seed = int(seed_s)
        net, rou = resolve(args.scenario, seed=seed)
        n, groups = probe_n_lanes(net, rou)
        n_dec = int(args.horizon // args.interval) + 2
        for kind in args.kinds.split(","):
            for rate_s in args.rates.split(","):
                rate = float(rate_s)
                mask = MaskScheduler(n, groups, n_dec, kind, rate,
                                     seed=1000 + seed, dur_decisions=30)
                actual = float(mask.z.mean())
                for spec in args.fills.split(","):
                    t0 = time.time()
                    m, recs = run_closed(net, rou, filler_spec=spec, mask=mask,
                                         horizon=args.horizon,
                                         interval=args.interval, seed=seed,
                                         trace=True)
                    a = audit_trace(recs)
                    row = dict(scenario=args.scenario, kind=kind, rate=rate,
                               actual=round(actual, 3), fill=spec, seed=seed,
                               wall_s=round(time.time() - t0, 1),
                               **{k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in m.items()},
                               **{f"aud_{k}": round(float(v.mean()), 4)
                                  for k, v in a.items()})
                    rows.append(row)
                    print(json.dumps(row))
    import csv
    tag = f"{args.scenario}_{int(time.time())}"
    path = os.path.join(out_dir, f"e0_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved:", path)


if __name__ == "__main__":
    main()
