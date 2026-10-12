# -*- coding: utf-8 -*-
"""Runtime adapter: drop-in replacement for RobustLight's Diffusion_Predictor.

Exposes demask_state(masked_state, action_onehot, long_state_con, mask) with the
same call signature used by utils/model_test.py, backed by a frozen ImputeFormer.

Node layout matches training: node = intersection_id * 12 + movement_slot.
The adapter maintains its own rolling window of (masked state, mask, applied
phase one-hot); phase one-hot = action+1, consistent with training's cur_phase
exog channel (cur_phase at step t equals the action applied at t-1).
"""
import numpy as np
import torch

from .if_model import ImputeFormerModel

N_INTER, N_MOVE, N_PHASE = 16, 12, 4


class IFRecoverer:
    def __init__(self, ckpt_path, device="cpu", autoreg=False):
        """autoreg=True mirrors RobustLight's long_state_con: the window history
        stores the model's own recovered values at missing positions, marked
        observed. Use only with checkpoints trained with --selfcond."""
        self.autoreg = autoreg
        ck = torch.load(ckpt_path, map_location=device)
        cfg = ck["config"]
        self.window = ck["window"]
        self.mu, self.sd = ck["scaler"]
        self.device = device
        self.model = ImputeFormerModel(
            num_nodes=N_INTER * N_MOVE, input_dim=1 + N_PHASE, output_dim=1,
            windows=self.window,
            input_embedding_dim=32, learnable_embedding_dim=96,
            feed_forward_dim=256, num_temporal_heads=4, num_layers=3,
            dim_proj=8, dropout=0.1).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.reset()

    def reset(self):
        self.buf_x = []      # each [N_INTER,N_MOVE] masked+scaled
        self.buf_mask = []   # each [N_INTER,N_MOVE] float 1=observed
        self.buf_u = []      # each [N_INTER,N_MOVE,N_PHASE] phase onehot

    def _push(self, x_masked, mask, action_oh):
        ph = torch.from_numpy(action_oh).float()                    # [16,4]
        u = ph.unsqueeze(1).expand(-1, N_MOVE, -1).contiguous()    # [16,12,4]
        self.buf_x.append(np.asarray(x_masked, dtype=np.float32))
        self.buf_mask.append(np.asarray(mask, dtype=np.float32))
        self.buf_u.append(u.numpy())
        if len(self.buf_x) > self.window:
            self.buf_x.pop(0)
            self.buf_mask.pop(0)
            self.buf_u.pop(0)

    @torch.no_grad()
    def demask_state(self, masked_state, action, states, mask, reverse_step=2):
        """Same signature as Diffusion_Predictor.demask_state.

        masked_state: [16,12] torch/np, corrupted features
        action:       [16,4] one-hot of the action just applied
        states:       ignored (long_state_con; history kept internally)
        mask:         [16,12] float, 1=observed
        returns:      np.ndarray [16,12] recovered feature
        """
        xm = np.asarray(masked_state.cpu() if torch.is_tensor(masked_state) else masked_state,
                        dtype=np.float32)
        mk = np.asarray(mask.cpu() if torch.is_tensor(mask) else mask, dtype=np.float32)
        act = np.asarray(action.cpu() if torch.is_tensor(action) else action, dtype=np.float32)
        self._push(xm, mk, act)

        # pad window by repeating the oldest frame (cold start)
        n = len(self.buf_x)
        xs = np.stack([self.buf_x[0]] * (self.window - n) + self.buf_x)      # [w,16,12]
        ms = np.stack([self.buf_mask[0]] * (self.window - n) + self.buf_mask)
        us = np.stack([self.buf_u[0]] * (self.window - n) + self.buf_u)      # [w,16,12,4]

        x = torch.tensor(xs.reshape(1, self.window, -1, 1), device=self.device)
        x = (x - self.mu) / self.sd
        m = torch.tensor(ms.reshape(1, self.window, -1, 1), device=self.device)
        u = torch.tensor(us.reshape(1, self.window, -1, N_PHASE), device=self.device)

        y = self.model(x, u, m) * self.sd + self.mu                          # [1,w,192,1]
        y_last = y[0, -1, :, 0].cpu().numpy().reshape(N_INTER, N_MOVE)

        out = xm.copy()
        miss = mk < 0.5
        out[miss] = y_last[miss]
        if self.autoreg:
            # rewrite latest buffered frame: history carries own predictions
            # at missing positions, marked observed (their long_state_con twin)
            self.buf_x[-1] = out.copy()
            self.buf_mask[-1] = np.ones_like(mk)
        return out
