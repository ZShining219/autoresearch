# -*- coding: utf-8 -*-
"""ImputeFormer backbone, vendored for dependency-light inference.

Model code is verbatim from ImputeFormer (tongnie/ImputeFormer, KDD'24):
  experiments/imputeformer/models/{ImputeFormer,Attention_layers}.py
Only the tsl-internal helpers (Dense/MLP/maybe_cat_exog) are re-implemented
locally so the module imports without the tsl/pytorch-lightning stack.
State-dict keys are identical to the upstream modules.
"""
import torch
import torch.nn as nn
from einops import repeat


def maybe_cat_exog(x, u, dim=-1):
    if u is not None:
        if u.dim() == 3:
            u = u.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        return torch.cat([x, u], dim=dim)
    return x


class Dense(nn.Module):
    def __init__(self, input_size, output_size, activation='linear', dropout=0., bias=True):
        super(Dense, self).__init__()
        act = {'relu': nn.ReLU, 'linear': nn.Identity}[activation]
        self.layer = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
            act(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None,
                 exog_size=None, n_layers=1, activation='relu', dropout=0.):
        super(MLP, self).__init__()
        if exog_size is not None:
            input_size += exog_size
        layers = [
            Dense(input_size=input_size if i == 0 else hidden_size,
                  output_size=hidden_size,
                  activation=activation,
                  dropout=dropout) for i in range(n_layers)
        ]
        self.mlp = nn.Sequential(*layers)
        if output_size is not None:
            self.readout = nn.Linear(hidden_size, output_size)
        else:
            self.register_parameter('readout', None)

    def forward(self, x, u=None):
        x = maybe_cat_exog(x, u)
        out = self.mlp(x)
        if self.readout is not None:
            return self.readout(out)
        return out


class AttentionLayer(nn.Module):
    """
    Multi-head scaled dot attention
    """
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    """
    Canonical self-attention layer
    """
    def __init__(self,
                 model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class EmbeddedAttention(nn.Module):
    """
    Spatial embedded attention layer
    """
    def __init__(self, model_dim, adaptive_embedding_dim):
        super().__init__()

        self.model_dim = model_dim

        self.FC_Q_K = nn.Linear(adaptive_embedding_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, value, emb):
        # V (batch_size, ..., seq_length, model_dim)
        # emb (..., length, model_dim)
        batch_size = value.shape[0]
        query = self.FC_Q_K(emb)
        key = self.FC_Q_K(emb)
        value = self.FC_V(value)

        # Q, K (..., length, model_dim)
        # V (batch_size, ..., length, model_dim)
        key = key.transpose(-1, -2)  # (..., model_dim, src_length)
        # attn_score = query @ key  # (..., tgt_length, src_length)
        # attn_score = torch.softmax(attn_score, dim=-1)
        # attn_score = repeat(attn_score, 'n s1 s2 -> b n s1 s2', b=batch_size)

        # re-normalization
        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)
        query = repeat(query, 'n s1 s2 -> b n s1 s2', b=batch_size)
        key = repeat(key, 'n s2 s1 -> b n s2 s1', b=batch_size)
        # re-normalization

        # out = attn_score @ value  # (batch_size, ..., tgt_length, model_dim)
        out = key @ value  # (batch_size, ..., tgt_length, model_dim)
        out = query @ out  # (batch_size, ..., tgt_length, model_dim)

        return out


class EmbeddedAttentionLayer(nn.Module):
    """
    Spatial embedded attention layer
    """
    def __init__(self,
                 model_dim, adaptive_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, seq_len, dim_proj, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(dim_proj, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(self.projector, 'dim_proj d_model -> repeat seq_len dim_proj d_model',
                           repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class ImputeFormerModel(nn.Module):
    """
    Spatiotempoarl Imputation Transformer
    """
    def __init__(
            self,
            num_nodes,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            learnable_embedding_dim=80,
            feed_forward_dim=256,
            num_temporal_heads=4,
            num_layers=3,
            dropout=0.,
            windows=24,
            dim_proj=10,
    ):
        super(ImputeFormerModel, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = windows
        self.out_steps = windows
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.learnable_embedding_dim = learnable_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + learnable_embedding_dim)
        self.num_temporal_heads = num_temporal_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.dim_proj = dim_proj

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(windows, num_nodes, learnable_embedding_dim)))

        self.readout = MLP(self.model_dim, self.model_dim, output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(self.num_nodes, self.dim_proj, self.model_dim, num_temporal_heads, self.model_dim, dropout)
             for _ in range(num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [EmbeddedAttentionLayer(self.model_dim, learnable_embedding_dim, feed_forward_dim)
             for _ in range(num_layers)])

    def forward(self, x, u, mask):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        # Whiten missing values
        x = x * mask

        x = maybe_cat_exog(x, u)
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        node_emb = self.learnable_embedding.expand(batch_size, *self.learnable_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        x = x.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        x = x.permute(0, 2, 1, 3)  # [b s n c]
        out = self.readout(x)

        return out
