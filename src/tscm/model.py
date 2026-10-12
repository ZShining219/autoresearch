"""M1: causal GRU + lane-movement-GNN probabilistic lane-count recovery.

Architecture (reimplementation of LMP-GNN A2 message passing over GRU-encoded
histories; GRIN GCGRU-style idea kept to a plain shared GRU for causality):

  per-lane input window (last T decisions, causal only):
      [log1p(Q_t), z_t, log1p(h_t), min(s_t,50)/50, log1p(vis_mean_t)]
      + static lane features + upstream-TLS phase one-hot (context)
  -> shared unidirectional GRU -> h_T per lane
  -> L lane-movement message-passing layers (LMP-GNN Eq.7-10)
  -> head: delta, log_var per lane; mu = ReLU(h_last + delta) (anchor, Eq.11)

Outputs per lane: predictive mean mu and marginal std sigma.
"""
import numpy as np
import torch
import torch.nn as nn


class LaneMoveMP(nn.Module):
    """LMP-GNN A2 layer (Eq.7-10): movement repr -> lane residual update.

    move_src / move_dst: [n_moves] lane-index lists are aggregated into
    (start_set, end_set) mean summaries per movement via sparse matmul
    prepared outside as torch sparse tensors:
        MS: [n_moves, n_lanes] 1/|S(r)| membership of start lanes
        MD: [n_moves, n_lanes] 1/|D(r)| membership of end lanes
        LS: [n_lanes, n_moves] 1/|{r: i in S(r)}| movements starting at lane i
        LD: [n_lanes, n_moves] 1/|{r: i in D(r)}| movements ending at lane i
    """

    def __init__(self, hid):
        super().__init__()
        self.Ws = nn.Linear(hid, hid)
        self.Wd = nn.Linear(hid, hid)
        self.Wself = nn.Linear(hid, hid)
        self.Wstart = nn.Linear(hid, hid)
        self.Wend = nn.Linear(hid, hid)

    def forward(self, h, MS, MD, LS, LD):
        # h: [B, L, H]; sparse mats applied via bmm-compatible einsum
        hs = torch.einsum("ml,blh->bmh", MS, h)   # movement start summary
        hd = torch.einsum("ml,blh->bmh", MD, h)   # movement end summary
        m = torch.relu(self.Ws(hs) + self.Wd(hd))              # [B, M, H]
        d = torch.einsum("lm,bmh->blh", LS, m)                 # lanes' start-movements
        u = torch.einsum("lm,bmh->blh", LD, m)                 # lanes' end-movements
        return h + torch.relu(self.Wself(h) + self.Wstart(d) + self.Wend(u))


class RecModel(nn.Module):
    """GRU + lane-movement GNN -> (mu, sigma) per lane."""

    def __init__(self, n_dyn, n_static, n_ctx, n_lanes, hid=64, n_layers=3,
                 n_phases=8):
        super().__init__()
        self.n_lanes, self.hid = n_lanes, hid
        self.enc = nn.Linear(n_dyn + n_static + n_ctx, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.mp = nn.ModuleList([LaneMoveMP(hid) for _ in range(n_layers)])
        self.head = nn.Linear(hid, 2)          # delta, raw log-var

    def forward(self, x_dyn, x_static, x_ctx, h_anchor, MS, MD, LS, LD):
        """x_dyn [B,T,L,Fd]; x_static [L,Fs]; x_ctx [B,T,L,Fc]; h_anchor [B,L]."""
        B, T, L, Fd = x_dyn.shape
        xs = x_static.unsqueeze(0).unsqueeze(0).expand(B, T, L, -1)
        xc = x_ctx
        x = torch.cat([x_dyn, xs, xc], -1)
        h0 = torch.relu(self.enc(x))                       # [B,T,L,H]
        h0 = h0.permute(0, 2, 1, 3).reshape(B * L, T, -1)  # lanes share GRU
        out, _ = self.gru(h0)
        h = out[:, -1].reshape(B, L, -1)                   # last-step hidden
        for layer in self.mp:
            h = layer(h, MS, MD, LS, LD)
        delta, lv = self.head(h).chunk(2, -1)
        lv = lv.squeeze(-1).clamp(-2.0, 6.0)               # log-var clip (Eq.11)
        mu = torch.relu(h_anchor + delta.squeeze(-1))
        sigma = torch.exp(0.5 * lv)
        return mu, sigma


# ---------------- graph construction ----------------

def build_movement_graph(world, lane_pos):
    """Movements = unique (from_edge -> to_edge) roadLinks across intersections.

    Returns torch sparse tensors MS, MD, LS, LD (see LaneMoveMP) and the
    movement list [(from_edge, to_edge)].
    """
    import scipy.sparse as sp
    moves = {}
    for it in world.intersections:
        for ll in it.phase_available_lanelinks:
            for fr, to in ll:
                key = (fr.rsplit("_", 1)[0], to.rsplit("_", 1)[0])
                moves.setdefault(key, [set(), set()])
                moves[key][0].add(lane_pos[fr])
                moves[key][1].add(lane_pos[to])
    move_keys = sorted(moves)
    n_l, n_m = len(lane_pos), len(move_keys)
    MS = np.zeros((n_m, n_l)); MD = np.zeros((n_m, n_l))
    LS = np.zeros((n_l, n_m)); LD = np.zeros((n_l, n_m))
    for k, (S, D) in enumerate((moves[k] for k in move_keys)):
        for i in S: MS[k, i] = 1.0 / len(S)
        for i in D: MD[k, i] = 1.0 / len(D)
        for i in S: LS[i, k] = 1.0
        for i in D: LD[i, k] = 1.0
    # normalize LS/LD over movements per lane
    LS /= np.maximum(LS.sum(1, keepdims=True), 1)
    LD /= np.maximum(LD.sum(1, keepdims=True), 1)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {"MS": t(MS), "MD": t(MD), "LS": t(LS), "LD": t(LD),
            "move_keys": move_keys, "n_moves": n_m}


# ---------------- static & context features ----------------

def lane_static_features(net, lane_index):
    """Per-lane static features (LMP-GNN Eq.5 style)."""
    import math
    feats = np.zeros((len(lane_index), 7), np.float32)
    for k, lid in enumerate(lane_index):
        lane = net.getLane(lid) if hasattr(net, "getLane") else None
        if lane is None:
            edge = net.getEdge(lid.rsplit("_", 1)[0])
            lane = edge.getLanes()[int(lid.rsplit("_", 1)[1])]
        length = lane.getLength(); vmax = lane.getSpeed()
        nlanes = len(lane.getEdge().getLanes())
        lidx = lane.getIndex()
        shp = lane.getShape()
        dx, dy = shp[-1][0] - shp[0][0], shp[-1][1] - shp[0][1]
        ang = math.atan2(dy, dx)
        width = lane.getWidth() if hasattr(lane, "getWidth") else 3.2
        feats[k] = [length / 1000.0, vmax / 20.0, width / 5.0,
                    math.sin(ang), math.cos(ang), lidx / 5.0, nlanes / 5.0]
    return feats


