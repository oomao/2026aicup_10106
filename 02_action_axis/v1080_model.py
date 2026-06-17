"""v1080 model: v1054 transformer encoder + multi-task heads, with a TRANSDUCTIVE
context projection REPLACING player-id embeddings (player-agnostic, OOV-safe).

Reuses the pretrained v1054 TransformerEncoder verbatim (loaded from v1054 ckpt).
The only structural change vs v1054 MultiTaskModel:
  - player_emb DISABLED (n_players=0 always).
  - a parallel nn.Sequential(Linear(16,d), GELU, LayerNorm, Dropout) projects the
    16-dim truncation-matched transductive vector; its output is concatenated into
    the pooled representation [last || mean || attn || ctx_e || transd_e] before the
    shared trunk. Rows with no prior get a zero vector + mp_has=0 (graceful OOV).

Encoder still sees only strokes 1..K (prefix); K+1 never enters. serverGetPoint
never input.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

# reuse v1054's encoder + data definitions
V1054_SRC = Path("E:/AICUP_O/models/v1054_serious_transformer/src")
sys.path.insert(0, str(V1054_SRC))
import data as D  # noqa: E402
from model import TransformerEncoder, AttnPool  # noqa: E402

N_TRANSDUCTIVE = 16


class TransductiveMultiTask(nn.Module):
    """Encoder + pooled prefix repr + transductive context -> action/point/server heads.

    pooled = [last-valid || masked-mean || attn-pool || ctx_proj(ctx) || transd_proj(transd)]
    No player embeddings (player-agnostic). transd is the 16-dim OOV-safe transductive vector.
    """

    def __init__(self, encoder: TransformerEncoder, n_action=15, n_point=10,
                 use_server=True, head_drop=0.3, transd_drop=0.2, d_transd=None):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.attn = AttnPool(d)
        self.ctx_proj = nn.Sequential(nn.Linear(D.N_CTX, d), nn.GELU())
        d_t = d_transd if d_transd is not None else d
        self.transd_proj = nn.Sequential(
            nn.Linear(N_TRANSDUCTIVE, d_t), nn.GELU(), nn.LayerNorm(d_t), nn.Dropout(transd_drop),
        )
        pooled_dim = d * 3 + d + d_t  # last+mean+attn (3d) + ctx (d) + transd (d_t)
        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(head_drop),
        )
        self.head_action = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d, n_action))
        self.head_point = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d, n_point))
        self.use_server = use_server
        if use_server:
            self.head_server = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d // 2, 1))

    def forward(self, field, num, lengths, ctx, transd):
        B, L, _ = field.shape
        ar = torch.arange(L, device=field.device).unsqueeze(0)
        valid = ar >= (L - lengths.unsqueeze(1))  # (B,L) True=valid (right-aligned)
        pad_mask = ~valid
        h = self.encoder(field, num, key_padding_mask=pad_mask)  # (B,L,d)
        last = h[:, -1, :]
        vm = valid.unsqueeze(-1).float()
        mean = (h * vm).sum(1) / vm.sum(1).clamp(min=1.0)
        attn = self.attn(h, valid)
        ctx_e = self.ctx_proj(ctx)
        transd_e = self.transd_proj(transd)
        pooled = torch.cat([last, mean, attn, ctx_e, transd_e], dim=1)
        z = self.shared(pooled)
        out_a = self.head_action(z)
        out_p = self.head_point(z)
        out_s = self.head_server(z).squeeze(-1) if self.use_server else None
        return out_a, out_p, out_s
