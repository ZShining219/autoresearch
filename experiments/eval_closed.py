"""E3/E4 closed-loop evaluation driver.

Loads a trained recovery checkpoint, wraps it in ModelAdapter, and runs
closed-loop episodes on held-out demand seeds under a chosen mask schedule.
Supports:
  --fill model:pred_mean | model:fixed_disc | model:staleness | heuristic fills
  --risk lcb / lcb_hold + --kappa / --gate   (M3 phase-level risk selection)
  baseline fills (zero/last/mean/road_mean/full) run without a model.

Outputs: results/closed/<tag>_<seed>.json + audit npz.
"""
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tscm.runner import run_closed, audit_trace
from tscm.scenarios import resolve
from tscm.obs import MaskScheduler, lane_groups_for
from tscm.world import SumoWorld
from tscm.infer import ModelAdapter, make_risk_fn
from tscm.model import RecModel
from collect_traces import load_meta

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
CKPT = os.path.join(os.path.dirname(__file__), "..", "results", "models")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "closed")


def load_model(ckpt_tag, meta, device):
    args = dict(np.load(os.path.join(CKPT, f"{ckpt_tag}_args.npy"),
                        allow_pickle=True).item())
    L = len(meta["lane_index"])
    model = RecModel(5, meta["static"].shape[1], meta["n_phases_max"] + 1, L,
                     hid=args["hid"], n_layers=args["layers"])
    sd = torch.load(os.path.join(CKPT, f"{ckpt_tag}.pt"), map_location=device,
                    weights_only=True)
    model.load_state_dict(sd["model"])
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="resco4x4")
    ap.add_argument("--ckpt", default=None, help="model tag; omit for heuristic fills")
    ap.add_argument("--fill", default="model:pred_mean")
    ap.add_argument("--risk", default="none", choices=["none", "lcb", "lcb_hold"])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--kind", default="correlated")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--dur", type=int, default=30)
    ap.add_argument("--seeds", default="9,10,11,12,13")
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    meta = load_meta(os.path.join(DATA, args.scenario, "meta.npz"))
    model, adapter = None, None
    if args.ckpt:
        model = load_model(args.ckpt, meta, args.device)
    risk_fn = (None if args.risk == "none" else
               make_risk_fn(kappa=args.kappa, gate=args.gate, mode=args.risk))
    tag = args.tag or (f"{args.ckpt or 'heur'}_{args.fill.replace(':', '-')}"
                       f"_{args.risk}{args.kappa}_{args.kind}{args.rate}")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in args.seeds.split(","):
        seed = int(s)
        net, rou = resolve(args.scenario, seed=seed)
        w = SumoWorld(net, rou).start()
        n, groups = len(w.lane_index), lane_groups_for(w)
        w.close()
        n_dec = int(args.horizon // args.interval) + 2
        mask = MaskScheduler(n, groups, n_dec, args.kind, args.rate,
                             seed=1000 + seed, dur_decisions=args.dur)
        if model is not None and args.fill.startswith("model"):
            adapter = ModelAdapter(model, meta, args.device)
        t0 = time.time()
        m, recs = run_closed(net, rou, filler_spec=args.fill, mask=mask,
                             model=adapter, decision_fn=risk_fn,
                             horizon=args.horizon, interval=args.interval,
                             seed=seed, trace=True)
        a = audit_trace(recs)
        row = dict(tag=tag, seed=seed, wall_s=round(time.time() - t0, 1),
                   mask_actual=round(float(mask.z.mean()), 3), **m,
                   **{f"aud_{k}": round(float(v.mean()), 4) for k, v in a.items()})
        rows.append(row)
        print(json.dumps(row))
        if args.trace:
            np.savez_compressed(
                os.path.join(OUT, f"trace_{tag}_s{seed}.npz"),
                q=np.stack([r["q"] for r in recs]),
                z=np.stack([r["z"] for r in recs]),
                x=np.stack([r["x"] for r in recs]))
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    mean = {f"{k}_mean": round(float(np.mean([r[k] for r in rows])), 4)
            for k in keys}
    std = {f"{k}_std": round(float(np.std([r[k] for r in rows])), 4)
           for k in keys}
    summ = dict(tag=tag, n_seeds=len(rows), **mean, **std)
    json.dump({"rows": rows, "summary": summ},
              open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print("SUMMARY:", json.dumps(summ))


if __name__ == "__main__":
    main()
