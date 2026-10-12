"""Training: M1 recovery model + M2 decision-aware objectives.

Objective (per batch):
    L = L_recon + beta * L_decision

  L_recon (masked positions only):
      'l1'  -> |mu - y|
      'nll' -> Gaussian NLL (LMP-GNN Eq.12 without constant)
  L_decision kinds (computed on completed x_hat = Q_last + z*mu):
      'press' -> MSE over phase scores  s = W_j x_hat vs s* = W_j y
      'rank'  -> CE with a* = argmax s*  (soft phase preservation)
      'spo'   -> SPO+ hinge for argmax: max_p(2s-s*)_p - (2s-s*)_{a*}
                 (adapted from Elmachtoub & Grigas via PyEPO formulation)
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from .model import RecModel


def _Ws_torch(meta, device):
    out = {}
    for tls, W in meta["Ws"].items():
        out[tls] = torch.tensor(W.toarray(), dtype=torch.float32, device=device)
    return out


def decision_losses(x_hat, y, z_any, Wt, legal, kind, tau=1.0):
    """Per-intersection decision losses on completed vectors.

    x_hat,y: [B,L]; z_any[tls]: whether intersection tls had any masked lane.
    Returns dict of scalar losses.
    """
    dev = x_hat.device
    press_mse, rank_ce, spo = [], [], []
    for tls, W in Wt.items():
        lg = legal[tls]
        Wl = W[lg]                                    # [P,L] legal only
        s = x_hat @ Wl.t()                            # [B,P]
        st = y @ Wl.t()
        a_star = st.argmax(-1)                        # first-max
        sel = z_any[tls]
        if sel.sum() == 0:
            continue
        s, st, a_star = s[sel], st[sel], a_star[sel]
        press_mse.append(Fn.mse_loss(s, st))
        rank_ce.append(Fn.cross_entropy(s / tau, a_star))
        hinge = (2 * s - st).max(-1).values - (2 * s - st).gather(
            1, a_star.unsqueeze(1)).squeeze(1)
        spo.append(hinge.mean())
    zero = torch.zeros((), device=dev)
    return {
        "press": torch.stack(press_mse).mean() if press_mse else zero,
        "rank": torch.stack(rank_ce).mean() if rank_ce else zero,
        "spo": torch.stack(spo).mean() if spo else zero,
    }


def masked_lanes_per_tls(meta):
    """Map tls -> lane indices used by its lanelinks (for z_any detection)."""
    m = {}
    pos = {l: k for k, l in enumerate(meta["lane_index"])}
    for tls, W in meta["Ws"].items():
        lanes = np.unique(W.nonzero()[1])
        m[tls] = torch.tensor(lanes, dtype=torch.long)
    return m


class Trainer:
    def __init__(self, meta, model, device="cpu", beta=0.0, recon="nll",
                 decision="none", tau=1.0, lr=1e-3, clip=5.0):
        self.meta, self.model, self.dev = meta, model.to(device), device
        self.beta, self.recon, self.decision, self.tau = beta, recon, decision, tau
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip = clip
        self.Wt = _Ws_torch(meta, device)
        self.mlanes = masked_lanes_per_tls(meta)
        self.static = torch.tensor(meta["static"], dtype=torch.float32, device=device)
        self.graph = {k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in meta["graph"].items()}

    def _forward(self, batch):
        g = self.graph
        mu, sigma = self.model(batch["dyn"].to(self.dev),
                               self.static,
                               batch["ctx"].to(self.dev),
                               batch["h_anchor"].to(self.dev),
                               g["MS"], g["MD"], g["LS"], g["LD"])
        return mu, sigma

    def step(self, batch):
        mu, sigma = self._forward(batch)
        z = batch["z_eval"].to(self.dev)
        y = batch["y"].to(self.dev)
        Q = batch["Q_last"].to(self.dev)
        zmask = z > 0.5
        if zmask.any():
            if self.recon == "nll":
                lv = 2 * torch.log(sigma.clamp_min(1e-4))
                recon = (0.5 * (lv + (y - mu) ** 2 / torch.exp(lv)))[zmask].mean()
            else:
                recon = (mu - y).abs()[zmask].mean()
        else:
            recon = torch.zeros((), device=z.device)
        x_hat = Q + z * mu
        loss = recon
        aux = {}
        if self.beta > 0 and self.decision != "none":
            z_any = {tls: (z[:, idx].sum(-1) > 0)
                     for tls, idx in ((k, v.to(self.dev)) for k, v in self.mlanes.items())}
            dl = decision_losses(x_hat, y, z_any, self.Wt, self.meta["legal"],
                                 self.decision, self.tau)
            loss = recon + self.beta * dl[self.decision]
            aux = {f"d_{k}": float(v) for k, v in dl.items()}
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        aux["recon"] = float(recon)
        return float(loss), aux

    @torch.no_grad()
    def predict(self, batch):
        mu, sigma = self._forward(batch)
        return mu, sigma

    def save(self, path):
        torch.save({"model": self.model.state_dict(),
                    "n_dyn": 5, "n_static": self.static.shape[-1],
                    "n_ctx": self.model.enc.in_features - 5 - self.static.shape[-1]},
                   path)


# ---------------- offline evaluation ----------------

@torch.no_grad()
def evaluate(model, trainer, dataset, device, batches=40, batch_size=256):
    """Masked-MAE/RMSE/NLL, coverage, score-MAE, disagree, regret vs argmax."""
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cnt_abs, cnt_sq, nll_v, cov1, cov2, n = [], [], [], [], [], 0
    scr_abs, dis, reg = [], [], []
    Wt = trainer.Wt; legal = trainer.meta["legal"]
    for bi, batch in enumerate(dl):
        if bi >= batches:
            break
        mu, sigma = trainer.predict(batch)
        z = batch["z_eval"].to(device) > 0.5
        y = batch["y"].to(device)
        Q = batch["Q_last"].to(device)
        if z.sum() == 0:
            continue
        err = (mu - y)[z]
        cnt_abs.append(err.abs()); cnt_sq.append(err ** 2)
        nll_v.append((torch.log(sigma) + 0.5 * (y - mu) ** 2 / sigma ** 2)[z])
        cov1.append((((y - mu).abs() <= sigma)[z]).float())
        cov2.append((((y - mu).abs() <= 2 * sigma)[z]).float())
        x_hat = Q + z.float() * mu
        for tls, W in Wt.items():
            lg = legal[tls]; Wl = W[lg]
            sel = (z[:, trainer.mlanes[tls].to(device)].sum(-1) > 0)
            if sel.sum() == 0:
                continue
            s = (x_hat @ Wl.t())[sel]; st = (y @ Wl.t())[sel]
            scr_abs.append((s - st).abs())
            a_t = st.argmax(-1); a_f = s.argmax(-1)
            dis.append((a_t != a_f).float())
            reg.append((st.gather(1, a_t.unsqueeze(1)) -
                        st.gather(1, a_f.unsqueeze(1))).squeeze(1))
        n += int(z.sum())
    cat = torch.cat
    return {
        "n_masked": n,
        "mae": float(cat(cnt_abs).mean()) if cnt_abs else 0.0,
        "rmse": float(cat(cnt_sq).mean().sqrt()) if cnt_sq else 0.0,
        "nll": float(cat(nll_v).mean()) if nll_v else 0.0,
        "cov1": float(cat(cov1).mean()) if cov1 else 0.0,
        "cov2": float(cat(cov2).mean()) if cov2 else 0.0,
        "score_mae": float(cat(scr_abs).mean()) if scr_abs else 0.0,
        "disagree": float(cat(dis).mean()) if dis else 0.0,
        "regret": float(cat(reg).mean()) if reg else 0.0,
    }
