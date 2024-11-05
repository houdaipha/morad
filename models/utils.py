from functools import partial
from dataclasses import dataclass, field
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

# -------------------- Transformer Head -------------------- #
@dataclass
class TransformerConfig:
    width: int = 768
    layers: int = 12
    heads: int = 12
    attn_dropout: int = 0.1
    mlp_dropout: int = 0.1
    attn_mask: torch.Tensor = None


@dataclass
class TransformerHeadConfig:
    frames: int
    width: int = 768
    layers: int = 12
    heads: int = 12
    attn_dropout: int = 0.1
    mlp_dropout: int = 0.1
    attn_mask: torch.Tensor = None
    out_features: int = 4
    dropout: float = 0.1
    out_pool: str = 'cls'


# from CLIP: https://github.com/openai/CLIP
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_dropout: int,
        mlp_dropout: int,
        attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            n_head,
            dropout=attn_dropout,
            batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(mlp_dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("o_dropout", nn.Dropout(mlp_dropout))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            config: TransformerConfig,):
        # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout =
        # 0.):
        super().__init__()
        self.width = config.width
        self.layers = config.layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(
                    d_model=config.width,
                    n_head=config.heads,
                    attn_dropout=config.attn_dropout,
                    mlp_dropout=config.mlp_dropout,
                    attn_mask=config.attn_mask) for _ in range(config.layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class TransformerHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        width = config.width
        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(config.frames + 1, width))
        self.ln_pre = LayerNorm(width)
        self.dropout = nn.Dropout(config.dropout)

        self.transformer = Transformer(
            config=TransformerConfig(
                width=width,
                layers=config.layers,
                heads=config.heads,
                attn_dropout=config.attn_dropout,
                mlp_dropout=config.mlp_dropout,
                attn_mask=config.attn_mask
            )
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Linear(width, config.out_features)

        if config.out_pool not in ('cls', 'avg', 'max', 'none'):
            raise ValueError(f'Invalid out_pool: {config.out_pool}')
        self.out_pool = config.out_pool

    def forward(self, x):
        cls_token = self.class_embedding.to(
            x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = self.dropout(x)

        x = self.transformer(x)

        if self.out_pool == 'cls':
            x = x[:, 0, :]
        elif self.out_pool == 'avg':
            x = x[:, 1:, :].mean(dim=1)
        elif self.out_pool == 'max':
            x, _ = torch.max(x[:, 1:, :], dim=1)
        elif self.out_pool == 'none':
            x = x[:, 1:, :]

        x = self.ln_post(x)
        x = self.proj(x)
        return x

# -------------------- Attention Pooling -------------------- #
class AttentionPoolConfig:
    frames: int
    embed_dim: int
    num_heads: int
    output_dim: int = None

# From CLIP: www.github.com/openai/CLIP
class AttentionPool(nn.Module):
    def __init__(self, config: AttentionPoolConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        self.positional_embedding = nn.Parameter(torch.randn(config.frames + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, config.output_dim or embed_dim)
        self.num_heads = config.num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NTW -> TNW
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (T+1)NW
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (T+1)NW
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


def main():
    config = TransformerHeadConfig(frames=100, width=768)
    head = TransformerHead(config)
    x = torch.randn(2, 100, 768)
    y = head(x)
    print(y.shape)


if __name__ == '__main__':
    main()
