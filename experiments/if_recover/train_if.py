#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train ImputeFormer on collected RobustLight/MaxPressure state traces.

Reuses the ImputeFormerModel backbone untouched; training loop is a thin
re-implementation of their imputer semantics:
  - structured mask S applied to a complete window (supervision targets)
  - whitening: extra Bernoulli hiding of observed inputs with p ~ {0.2,0.5,0.8}
  - loss = L1(y_hat, y_true | S|W positions) + f1_w * Freg (freq regularizer)
Input: x channel = pressure_queue (whitened), u = cur_phase one-hot (not masked).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, "/data/users/zfh/workspace/projects/auto_research/refs/ImputeFormer/experiments")

import if_data  # noqa: E402
from imputeformer.models.ImputeFormer import ImputeFormerModel  # noqa: E402


def freg(y_hat, y, mask):
    """Frequency regularizer from ImputeFormerImputer (mask=True -> eval positions)."""
    y_tilde = torch.where(mask.bool().unsqueeze(-1), y_hat, y)
    y_f = torch.fft.fftn(y_tilde)
    y_f = y_f.reshape(y_f.shape[0], -1)
    return torch.mean(torch.sum(torch.abs(y_f), dim=1) / y_f.numel())


def masked_l1(y_hat, y, mask):
    m = mask.bool().unsqueeze(-1)
    if not m.any():
        return torch.tensor(0.0, device=y.device)
    return F.l1_loss(y_hat[m.expand_as(y)], y[m.expand_as(y)])


def forward(model, xw, uw, obs_mask):
    # xw [b,w,192,1] values; obs_mask [b,w,192] True=observed
    x = xw * obs_mask.unsqueeze(-1).float()
    return model(x, uw, obs_mask.unsqueeze(-1).float())


def selfcond_forward(model, xb, ub, hide):
    """Two-pass forward mimicking deployment-time autoregressive feedback.

    pass1: missing positions are zero throughout -> y1
    pass2: history frames (0..w-2) get y1 at hidden positions marked OBSERVED
           (mirrors RobustLight's long_state_con holding its own predictions);
           last frame keeps true missingness -> y2 (the deployed output).
    hide: [b,w,192] bool True=hidden
    """
    obs = ~hide
    with torch.no_grad():
        y1 = forward(model, xb, ub, obs)
    h3 = hide.unsqueeze(-1)
    x2 = torch.where(h3, y1, xb)
    obs2 = torch.ones_like(obs)
    obs2[:, -1] = obs[:, -1]
    y2 = model(x2 * obs2.unsqueeze(-1).float(), ub, obs2.unsqueeze(-1).float())
    return y2


def evaluate(model, Xv, Uv, scaler, eval_masks, device, bs=64, selfcond=False):
    mu, sd = scaler
    model.eval()
    losses = {}
    with torch.no_grad():
        for name, masks in eval_masks.items():
            tot, cnt = 0.0, 0
            for i in range(0, len(Xv), bs):
                xb = torch.tensor((Xv[i:i + bs] - mu) / sd, device=device)
                ub = torch.tensor(Uv[i:i + bs], device=device)
                mb = torch.tensor(masks[i:i + bs], device=device)
                if selfcond:
                    yh = selfcond_forward(model, xb, ub, mb)
                    tgt = mb[:, -1:]
                    yh = yh[:, -1:]
                    xb_c = xb[:, -1:]
                else:
                    obs = ~mb
                    yh = forward(model, xb, ub, obs)
                    tgt, xb_c = mb, xb
                tot += masked_l1(yh, xb_c, tgt).item() * len(xb)
                cnt += len(xb)
            losses[name] = tot / max(cnt, 1)
    model.train()
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/if_recover/traces")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f1-w", type=float, default=0.01)
    ap.add_argument("--out", default="results/if_recover/model_if.pt")
    ap.add_argument("--train-pat", default="*var0[0-3]*.npz")
    ap.add_argument("--val-pat", default="*var04[4-7].npz")
    ap.add_argument("--extra-val", default="orig_5816.npz")
    ap.add_argument("--selfcond", action="store_true",
                    help="align w/ RobustLight: history holds own predictions; "
                         "loss on last frame only (dynamics-style target)")
    ap.add_argument("--aux-w", type=float, default=0.3,
                    help="weight of whole-window aux L1 when --selfcond")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "/data/users/zfh/workspace/projects/auto_research"
    traces = os.path.join(root, args.traces)

    # ---------------- data ----------------
    Xtr, Utr = if_data.build_window_bank(traces, args.window, [args.train_pat])
    Xv, Uv = if_data.build_window_bank(traces, args.window, [args.val_pat])
    Xo, Uo = if_data.build_window_bank(traces, args.window, [args.extra_val])
    scaler = if_data.fit_scaler(Xtr)
    print("train windows:", Xtr.shape, "val:", Xv.shape, "orig:", Xo.shape)

    # held-out eval masks = eval-like direction damage (fixed, shared)
    def eval_mask_set(X):
        B = X.shape[0]
        m25 = np.stack([if_data.mask_dir_global(rng, args.window, directions=["W"])
                        for _ in range(B)])
        m50 = np.stack([if_data.mask_dir_global(rng, args.window, directions=["W", "E"])
                        for _ in range(B)])
        mp = np.stack([if_data.mask_point(rng, args.window, p=0.25) for _ in range(B)])
        return {"dirW": m25, "dirWE": m50, "point25": mp}

    eval_masks_v = eval_mask_set(Xv)
    eval_masks_o = eval_mask_set(Xo)

    # ---------------- model ----------------
    model = ImputeFormerModel(num_nodes=if_data.N_NODE, input_dim=1 + if_data.N_PHASE,
                              output_dim=1, windows=args.window,
                              input_embedding_dim=32, learnable_embedding_dim=96,
                              feed_forward_dim=256, num_temporal_heads=4,
                              num_layers=3, dim_proj=8, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mu, sd = scaler

    best = np.inf
    bad = 0
    os.makedirs(os.path.dirname(os.path.join(root, args.out)), exist_ok=True)
    hist = []
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb_np = (Xtr[idx] - mu) / sd
            xb = torch.tensor(xb_np, device=device)
            ub = torch.tensor(Utr[idx], device=device)
            B = len(idx)
            S = np.stack([if_data.sample_train_mask(rng, args.window)[0] for _ in range(B)])
            whiten_p = rng.choice([0.2, 0.5, 0.8], size=(B, 1, 1))
            W = rng.rand(B, args.window, if_data.N_NODE) < whiten_p
            hide = torch.tensor(S | W, device=device)          # True = hidden
            obs = ~hide
            S_dev = torch.tensor(S, device=device)
            if args.selfcond:
                yh = selfcond_forward(model, xb, ub, hide)
                loss = masked_l1(yh[:, -1:], xb[:, -1:], hide[:, -1:])
                loss = loss + args.aux_w * masked_l1(yh, xb, hide)
            else:
                yh = forward(model, xb, ub, obs)
                loss = masked_l1(yh, xb, hide)
            loss = loss + args.f1_w * freg(yh, xb, S_dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * B
            cnt += B
        sched.step()
        vl = evaluate(model, Xv, Uv, scaler, eval_masks_v, device, bs=args.bs,
                      selfcond=args.selfcond)
        vo = evaluate(model, Xo, Uo, scaler, eval_masks_o, device, bs=args.bs,
                      selfcond=args.selfcond)
        row = {"ep": ep, "train": tot / cnt,
               **{("v_" + k): v for k, v in vl.items()},
               **{("o_" + k): v for k, v in vo.items()}}
        hist.append(row)
        print(row)
        crit = vl["dirW"] + vl["dirWE"]
        if crit < best:
            best = crit
            bad = 0
            torch.save({"model": model.state_dict(), "scaler": scaler,
                        "window": args.window, "config": vars(args)},
                       os.path.join(root, args.out))
        else:
            bad += 1
            if bad >= 15:
                print("early stop")
                break
    with open(os.path.join(root, "results/if_recover/train_hist.json"), "w") as f:
        json.dump(hist, f)
    print("best val(dirW+dirWE):", best)


if __name__ == "__main__":
    main()
