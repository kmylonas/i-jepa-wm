from torch import nn
import torch

class LayerNorm(nn.Module):
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, X):
        'X: (B, N, D)'
        mu = torch.mean(X, dim=-1, keepdim=True)
        var = ((X - mu)**2).mean(dim=-1, keepdim=True)

        X_norm = (X - mu) / torch.sqrt(var + self.eps)

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
    def __init__(self, embed_dim, num_heads):
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


        nn.init.trunc_normal_(self.W_q, std=0.02)
        nn.init.trunc_normal_(self.W_k, std=0.02)
        nn.init.trunc_normal_(self.W_v, std=0.02)
        nn.init.trunc_normal_(self.W_o, std=0.02)


    def forward(self, X):

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

        attn_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, V) #(B, num_heads, N, head_dim)
        out = out.transpose(2, 1) #(B, N, num_heads, head_dim)
        out = out.reshape(batch_size, num_tokens, embed_dim)

        out = torch.matmul(out, self.W_o) + self.b_o

        return out





class MLP(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        layers_list = []
        for l in range(num_layers):
            layers_list.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers_list.pop()
        self.mlp = nn.Sequential(*layers_list)

    def forward(self, X):
        return self.mlp(X)




class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # self.head_dim = embed_dim / num_heads

        # self.attn_heads = [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        self.mlp = MLP(3, embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, X):
        
        X = self.mha(self.ln1(X)) + X
        X = self.mlp(self.ln2(X)) + X
        return X




class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, images):
        # images: [B, C, H, W]
        # breakpoint()

        x = self.projection(images)
        # [B, embed_dim, grid_height, grid_width]

        x = x.flatten(start_dim=2)
        # [B, embed_dim, number_of_patches]

        x = x.transpose(1, 2)
        # [B, number_of_patches, embed_dim]

        return x



class ViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_t_blocks, embed_dim, num_heads, num_classes):
        super().__init__()

        self.cls = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, embed_dim)) #64 image patches + 1 for cls

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.patch_embedding = PatchEmbedding(patch_size, 3, embed_dim)

        # self.emb_proj = nn.Linear(patch_dim, embed_dim)
        self.tb1 = TransformerBlock(embed_dim, num_heads)
        self.tb2 = TransformerBlock(embed_dim, num_heads)
        self.final_norm = LayerNorm(embed_dim)
        self.classification_head = nn.Linear(embed_dim, num_classes)


    def forward(self, X):
        # X = self.emb_proj(X)
        batch_size = X.shape[0]
        cls_exp = self.cls.expand(batch_size, -1, -1) # (B, 1, emb_dim)

        X = self.patch_embedding(X) 
        X = torch.cat([cls_exp, X], dim=1) # (B, 65, emb_dim)

        X = X + self.pos_embed # broadcasting will take care of this


        X = self.tb1(X)
        X = self.tb2(X)

        X = self.final_norm(X)
        logits = self.classification_head(X[:, 0]) #CLS used for classification
        return logits






        
