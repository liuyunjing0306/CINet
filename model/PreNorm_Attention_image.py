import torch
from torch import nn
from einops import rearrange

## 图像 文本 CharFormer 空间展平
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


# CROSS-ATTENTION
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # for key and value

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, image_embed, com_embed):
        # image_embed as query, com_embed as key and value
        q = self.to_q(image_embed)
        k, v = self.to_kv(com_embed).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Calculate attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # Apply attention to value
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer with Cross-Attention
class ImageAttention(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            )

    def forward(self, image_features, com_features):
        # Flatten image and com features
        image_embed = rearrange(image_features, 'b c h w -> b (h w) c')  # [batch_size, 64, 1024]
        com_embed = rearrange(com_features, 'b c h w -> b (h w) c')  # [batch_size, 64, 1024]

        # Apply cross-attention for each layer
        for attn in self.layers:
            image_embed = attn(image_embed, com_embed) + image_embed  # Cross-Attention with residual connection

        # Reshape back to original spatial dimensions
        image_embed = rearrange(image_embed, 'b (h w) c -> b c h w', h=8, w=8)  # [batch_size, 1024, 8, 8]

        return image_embed



