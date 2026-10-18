"""Action-conditioned I-JEPA world model with proprioceptive state."""

import torch
from torch import nn

from src.mylonas_ijepa_wm.vit import (
    LayerNorm,
    TransformerBlock,
    create_block_causal_mask,
)


class ProprioViT(nn.Module):
    """Predict next visual tokens and normalized (x, y, vx, vy)."""

    def __init__(
        self,
        num_patches,
        num_hist,
        num_t_blocks,
        ijepa_dim,
        action_dim,
        action_embed_dim,
        proprio_dim,
        proprio_embed_dim,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_hist = num_hist
        self.ijepa_dim = ijepa_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.GELU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, proprio_embed_dim),
            nn.GELU(),
            nn.Linear(proprio_embed_dim, proprio_embed_dim),
        )
        self.input_projection = nn.Linear(
            ijepa_dim + action_embed_dim + proprio_embed_dim,
            embed_dim,
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.empty(1, 1, num_patches, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.empty(1, num_hist, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_t_blocks)
        ])
        self.register_buffer(
            "attn_mask",
            create_block_causal_mask(num_hist, num_patches),
            persistent=False,
        )
        self.final_norm = LayerNorm(embed_dim)
        self.visual_head = nn.Linear(embed_dim, ijepa_dim)
        self.proprio_head = nn.Linear(embed_dim, proprio_dim)

    def forward(self, latents, states, actions):
        if latents.ndim != 4 or latents.shape[1:3] != (
            self.num_hist,
            self.num_patches,
        ) or latents.shape[-1] != self.ijepa_dim:
            raise ValueError("latents have the wrong shape")
        if states.shape != (
            latents.shape[0], self.num_hist, self.proprio_dim
        ):
            raise ValueError("states have the wrong shape")
        if actions.shape != (
            latents.shape[0], self.num_hist, self.action_dim
        ):
            raise ValueError("actions have the wrong shape")

        action_tokens = self.action_encoder(actions).unsqueeze(2).expand(
            -1, -1, self.num_patches, -1
        )
        proprio_tokens = self.proprio_encoder(states).unsqueeze(2).expand(
            -1, -1, self.num_patches, -1
        )
        tokens = torch.cat([latents, action_tokens, proprio_tokens], dim=-1)
        tokens = self.input_projection(tokens)
        tokens = tokens + self.spatial_pos_embed + self.temporal_pos_embed

        batch_size = tokens.shape[0]
        tokens = tokens.reshape(
            batch_size, self.num_hist * self.num_patches, -1
        )
        for block in self.transformer_blocks:
            tokens = block(tokens, attn_mask=self.attn_mask)
        tokens = self.final_norm(tokens)
        tokens = tokens.reshape(
            batch_size, self.num_hist, self.num_patches, -1
        )
        return self.visual_head(tokens), self.proprio_head(tokens.mean(dim=2))
