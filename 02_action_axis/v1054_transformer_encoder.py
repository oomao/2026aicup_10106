"""Serious transformer stroke encoder + MSM pretrain head + multi-task heads.

Differs from the FAILED versions:
  - v104/v105/v107: from-scratch / external-BM-contaminated / killed early. Here:
    NO external data, HEAVY clean MSM pretrain to convergence, NO premature kill.
  - v1040 ShuttleNet: dual-encoder + TAA + gated-fusion junk that the isolating
    ablation proved HURTS at 15k. Here: a single clean transformer encoder, no
    fusion, no role-split sub-encoders.
  - v1021 BiLSTM: encoder is a transformer (self-attention handles all K lengths
    uniformly) and pretrain is heavy. Multi-head pooling instead of last+mean only.

Right-sized: d_model=160, 4 layers, 8 heads, ffn=4x, heavy dropout + weight decay.
Causal masking is implicit: only strokes 1..K (prefix) are ever fed; K+1 never
enters. Padding mask handles left-pad.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

import data as D


class StrokeEmbed(nn.Module):
    """Embed a stroke: sum of per-field categorical embeds + numeric projection."""

    def __init__(self, d_model, emb_drop=0.1):
        super().__init__()
        self.fields = D.FIELD_LIST
        self.embs = nn.ModuleDict({
            f: nn.Embedding(D.VOCAB[f], d_model, padding_idx=D.PAD) for f in self.fields
        })
        self.num_proj = nn.Sequential(nn.Linear(D.N_NUM, d_model), nn.GELU())
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(emb_drop)
        self.d_model = d_model

    def forward(self, field, num):
        x = 0
        for j, f in enumerate(self.fields):
            x = x + self.embs[f](field[:, :, j])
        x = x + self.num_proj(num)
        return self.drop(self.ln(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Pre-LN transformer encoder over stroke tokens. Returns per-stroke states."""

    def __init__(self, d_model=160, n_layers=4, n_heads=8, ffn_mult=4,
                 dropout=0.2, emb_drop=0.1):
        super().__init__()
        self.embed = StrokeEmbed(d_model, emb_drop)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model
        self.d_model = d_model

    def forward(self, field, num, key_padding_mask=None):
        # key_padding_mask: (B, L) bool, True where PAD (ignored by attention)
        x = self.embed(field, num)
        x = self.pe(x)
        h = self.enc(x, src_key_padding_mask=key_padding_mask)
        return self.norm(h)


class MSMHead(nn.Module):
    """Masked-stroke-modeling heads: reconstruct masked tokens for 5 fields."""

    MASK_FIELDS = ["actionId", "pointId", "spinId", "strengthId", "positionId"]

    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.LayerNorm(in_dim))
        self.heads = nn.ModuleDict({f: nn.Linear(in_dim, D.VOCAB[f]) for f in self.MASK_FIELDS})

    def forward(self, h):
        z = self.proj(h)
        return {f: self.heads[f](z) for f in self.MASK_FIELDS}


class AttnPool(nn.Module):
    """Learned attention pooling over valid (non-pad) prefix positions."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, h, valid_mask):
        # h: (B,L,d) ; valid_mask: (B,L) bool True=valid
        score = self.q(h).squeeze(-1)  # (B,L)
        score = score.masked_fill(~valid_mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)  # (B,L,1)
        return (h * w).sum(1)  # (B,d)


class MultiTaskModel(nn.Module):
    """Encoder + pooled prefix repr -> action + point (+ optional server) heads.

    Pooled = [last-valid-token state || masked-mean || attn-pool] (3*d) concat with
    the OOV-safe target context (parity of K+1 striker, sex) projected to d, and
    OPTIONALLY learnable player embeddings for the K+1 striker + opponent (index 0
    reserved for UNKNOWN/OOV). Player embeds carry heavy dropout so the model does
    not over-rely on them (graceful OOV degradation -> falls back to the encoder
    representation). n_players = 0 disables player embeddings (player-agnostic).
    """

    def __init__(self, encoder: TransformerEncoder, n_action=15, n_point=10,
                 use_server=True, head_drop=0.3, n_players=0, d_player=48,
                 player_emb_drop=0.3):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.attn = AttnPool(d)
        self.ctx_proj = nn.Sequential(nn.Linear(D.N_CTX, d), nn.GELU())
        self.n_players = n_players
        extra = 0
        if n_players > 0:
            # +1 for the reserved UNKNOWN index 0
            self.player_emb = nn.Embedding(n_players + 1, d_player, padding_idx=0)
            self.player_drop = nn.Dropout(player_emb_drop)
            extra = d_player * 2  # self + opponent
        pooled_dim = d * 3 + d + extra
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(head_drop),
        )
        self.head_action = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d, n_action))
        self.head_point = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d, n_point))
        self.use_server = use_server
        if use_server:
            self.head_server = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d // 2, 1))

    def forward(self, field, num, lengths, ctx, pid_self=None, pid_opp=None):
        B, L, _ = field.shape
        ar = torch.arange(L, device=field.device).unsqueeze(0)
        valid = ar >= (L - lengths.unsqueeze(1))  # (B,L) True=valid (right-aligned)
        pad_mask = ~valid  # True where PAD
        h = self.encoder(field, num, key_padding_mask=pad_mask)  # (B,L,d)
        last = h[:, -1, :]  # most recent stroke (always valid when lengths>=1)
        vm = valid.unsqueeze(-1).float()
        mean = (h * vm).sum(1) / vm.sum(1).clamp(min=1.0)
        attn = self.attn(h, valid)
        ctx_e = self.ctx_proj(ctx)
        parts = [last, mean, attn, ctx_e]
        if self.n_players > 0 and pid_self is not None:
            pe_s = self.player_drop(self.player_emb(pid_self))
            pe_o = self.player_drop(self.player_emb(pid_opp))
            parts += [pe_s, pe_o]
        pooled = torch.cat(parts, dim=1)
        z = self.shared(pooled)
        out_a = self.head_action(z)
        out_p = self.head_point(z)
        out_s = self.head_server(z).squeeze(-1) if self.use_server else None
        return out_a, out_p, out_s
