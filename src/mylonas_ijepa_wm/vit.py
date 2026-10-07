from torch import nn
import torch


def create_block_causal_mask(num_hist: int, num_patches: int):
    frame_mask = torch.tril(
        torch.ones(num_hist, num_hist, dtype=torch.bool)
    )

    return frame_mask.repeat_interleave(
        num_patches,
        dim=0,
    ).repeat_interleave(
        num_patches,
        dim=1,
    )


class LayerNorm(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, X):
        'X: (B, N, D)'
        input_dtype = X.dtype
        X = X.float()

        mu = torch.mean(X, dim=-1, keepdim=True)
        var = ((X - mu)**2).mean(dim=-1, keepdim=True)

        X_norm = (X - mu) / torch.sqrt(var + self.eps)
        X_norm = X_norm.to(dtype=input_dtype)

        return self.gamma * X_norm + self.beta


class AttentionHead(nn.Module):
    # Not used -- only for educational purposes. Use MHA instead
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = embed_dim

        self.W_q = nn.Parameter(torch.empty(embed_dim, head_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, head_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, head_dim))

        nn.init.trunc_normal_(self.W_q, std=0.02)
        nn.init.trunc_normal_(self.W_k, std=0.02)
        nn.init.trunc_normal_(self.W_v, std=0.02)
    

    def forward(self, X):
        Q = torch.matmul(X, self.W_q)
        K = torch.matmul(X, self.W_k)
        V = torch.matmul(X, self.W_v)

        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / torch.sqrt(torch.tensor(self.head_dim))

        attn_mat = torch.softmax(scores, dim= -1)

        out = torch.matmul(attn_mat, V)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.b_q = nn.Parameter(torch.zeros(embed_dim))
        self.b_k = nn.Parameter(torch.zeros(embed_dim))
        self.b_v = nn.Parameter(torch.zeros(embed_dim))

        self.W_o = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_o = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(dropout)


        nn.init.trunc_normal_(self.W_q, std=0.02)
        nn.init.trunc_normal_(self.W_k, std=0.02)
        nn.init.trunc_normal_(self.W_v, std=0.02)
        nn.init.trunc_normal_(self.W_o, std=0.02)


    def forward(self, X, attn_mask=None):

        batch_size, num_tokens, embed_dim = X.shape

        Q = torch.matmul(X, self.W_q) + self.b_q  # (B, N, embed_dim)
        K = torch.matmul(X, self.W_k) + self.b_k
        V = torch.matmul(X, self.W_v) + self.b_v

        #(B, N, num_head, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.reshape( 
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        K = K.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        V = V.reshape(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim
        ).transpose(2,1)

        scores = torch.matmul(Q, K.transpose(-2,-1))
        scores = scores * self.scale

        if attn_mask is not None:
            scores = scores.masked_fill(
                ~attn_mask,
                float("-inf"),
            )

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V) #(B, num_heads, N, head_dim)
        out = out.transpose(2, 1) #(B, N, num_heads, head_dim)
        out = out.reshape(batch_size, num_tokens, embed_dim)

        out = torch.matmul(out, self.W_o) + self.b_o
        out = self.dropout(out)

        return out





class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.mlp(X)




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.head_dim = embed_dim / num_heads

        # self.attn_heads = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, X, attn_mask=None):
        
        X = self.mha(self.ln1(X), attn_mask=attn_mask) + X
        X = self.mlp(self.ln2(X)) + X
        return X




class ViT(nn.Module):
    def __init__(
        self,
        num_patches,
        num_hist,
        num_t_blocks,
        ijepa_dim,
        action_dim,
        action_embed_dim,
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
        self.action_embed_dim = action_embed_dim
        self.embed_dim = embed_dim

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.GELU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

        self.input_projection = nn.Linear(
            ijepa_dim + action_embed_dim,
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
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                dropout,
            )
            for _ in range(num_t_blocks)
        ])

        self.register_buffer(
            "attn_mask",
            create_block_causal_mask(num_hist, num_patches),
            persistent=False,
        )

        self.final_norm = LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, ijepa_dim)

    def forward(self, X, actions):
        action_embeddings = self.action_encoder(actions)
        action_embeddings = action_embeddings.unsqueeze(2).expand(
            -1,
            -1,
            self.num_patches,
            -1,
        )

        X = torch.cat([X, action_embeddings], dim=-1)
        X = self.input_projection(X)
        X = X + self.spatial_pos_embed + self.temporal_pos_embed

        batch_size = X.shape[0]
        X = X.reshape(
            batch_size,
            self.num_hist * self.num_patches,
            self.embed_dim,
        )

        for block in self.transformer_blocks:
            X = block(X, attn_mask=self.attn_mask)

        X = self.final_norm(X)
        X = X.reshape(
            batch_size,
            self.num_hist,
            self.num_patches,
            self.embed_dim,
        )

        X = self.output_projection(X)
        return X






        
