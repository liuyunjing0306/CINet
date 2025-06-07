import torch
from torch import nn
from einops import rearrange


## 图像 文本 CharFormer 通道展平


# 为了对文本编码器的输出（128，2048）进行处理，在进行交叉注意力之前先进行线性映射+重塑形状。

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # self.relu = nn.ReLU()

    def forward(self, x):
        return self.proj(x)


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

    def forward(self, image_embed, text_embed):
        # image_embed as query, text_embed as key and value
        q = self.to_q(image_embed)
        k, v = self.to_kv(text_embed).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Calculate attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # Apply attention to value
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer with Cross-Attention
class CrossAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            )
        self.generator_proj = Generator(2048, 1024 * 64)

    def forward(self, image_features, text_features):
        # Flatten image and text features
        image_embed = rearrange(image_features, 'b c h w -> b c (h w)')  # 修改位置: [batch_size, 1024, 64]
        # 添加线性层
        text_embed_proj=self.generator_proj(text_features)

        text_embed = rearrange(text_embed_proj, 'b (c h) -> b c h',c=1024,h=64)  # 修改位置: [batch_size, 1024, 64]

        # Apply cross-attention for each layer
        for attn in self.layers:
            image_embed = attn(image_embed, text_embed) + image_embed  # Cross-Attention with residual connection

        # Reshape back to original spatial dimensions
        image_embed = rearrange(image_embed, 'b c (h w) -> b c h w', h=8, w=8)  # [batch_size, 1024, 8, 8]

        return image_embed




