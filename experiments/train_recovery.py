"""Train/evaluate M1 recovery model with optional M2 decision-aware loss.

Usage example:
  python train_recovery.py --scenario resco4x4 --train-seeds 1,2,3 \
      --val-seeds 4 --decision press --beta 0.5 --epochs 10
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tscm.data import TraceDataset
from tscm.model import RecModel
from tscm.train import Trainer, evaluate
from collect_traces import load_meta

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "traces")
CKPT = os.path.join(os.path.dirname(__file__), "..", "results", "models")


def load_episodes(scenario, seeds):
    eps = []
    for s in seeds:
        z = np.load(os.path.join(DATA, scenario, f"ep_seed{s}.npz"))
        eps.append({"q_true": z["q_true"], "z_sched": z["z_sched"],
                    "phase_idx": z["phase_idx"]})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="resco4x4")
    ap.add_argument("--train-seeds", default="1,2,3")
    ap.add_argument("--val-seeds", default="4")
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--pmax", type=float, default=0.5)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--recon", default="nll", choices=["nll", "l1"])
    ap.add_argument("--decision", default="none",
                    choices=["none", "press", "rank", "spo"])
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    meta = load_meta(os.path.join(DATA, args.scenario, "meta.npz"))
    L = len(meta["lane_index"])
    n_ph = meta["n_phases_max"]
    model = RecModel(n_dyn=5, n_static=meta["static"].shape[1],
                     n_ctx=n_ph + 1, n_lanes=L, hid=args.hid,
                     n_layers=args.layers, n_phases=n_ph)
    tr_ep = load_episodes(args.scenario, [int(s) for s in args.train_seeds.split(",")])
    va_ep = load_episodes(args.scenario, [int(s) for s in args.val_seeds.split(",")])
    tr_ds = TraceDataset(tr_ep, meta, window=args.window, pmax=args.pmax, seed=0)
    va_ds = TraceDataset(va_ep, meta, window=args.window, pmax=args.pmax, seed=1)
    from torch.utils.data import DataLoader
    dl = DataLoader(tr_ds, batch_size=args.bs, shuffle=True)
    trainer = Trainer(meta, model, device=args.device, beta=args.beta,
                      recon=args.recon, decision=args.decision, lr=args.lr)
    print(f"dataset: {len(tr_ds)} train / {len(va_ds)} val samples; "
          f"{sum(p.numel() for p in model.parameters())} params; device={args.device}")
    best = None
    for ep in range(args.epochs):
        t0, tot, k = time.time(), 0.0, 0
        model.train()
        for batch in dl:
            loss, _ = trainer.step(batch)
            tot += loss; k += 1
        model.eval()
        m = evaluate(model, trainer, va_ds, args.device, batches=12)
        print(f"ep{ep}: train_loss={tot/k:.4f} "
              f"val MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} cov1={m['cov1']:.2f} "
              f"scoreMAE={m['score_mae']:.3f} dis={m['disagree']*100:.1f}% "
              f"reg={m['regret']:.3f} ({time.time()-t0:.0f}s)")
        if best is None or m["mae"] < best["mae"]:
            best = m
            os.makedirs(CKPT, exist_ok=True)
            tag = args.tag or f"{args.scenario}_d{args.decision}_b{args.beta}"
            path = os.path.join(CKPT, f"{tag}.pt")
            trainer.save(path)
            np.save(os.path.join(CKPT, f"{tag}_args.npy"), vars(args),
                    allow_pickle=True)
    print("BEST:", json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
