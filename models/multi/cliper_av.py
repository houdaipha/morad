import sys
import logging
import json
import yaml
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import wav2vec2_model
from einops import rearrange, repeat
from models.utils import TransformerConfig, Transformer
from models.utils import TransformerHeadConfig, TransformerHead

logger = logging.getLogger(__name__)

@dataclass
class VisionTransformerConfig:
    width: int = 768
    heads: int = 12
    layers: int = 12
    patch_size: int = 16
    grid_size: int = 14
    image_resolution: int = 224
    output_dim: int = 512

@dataclass
class CLIPTransformerConfig:
    transformer_head: TransformerHeadConfig
    pretrained_weights: str
    proj_dropout: float = 0.1
    freeze: bool = True

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)

@dataclass
class WhisperConfig:
    pretrained_weights: str
    freeze: bool
    proj_dropout: float
    transformer_head: TransformerHeadConfig

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)

@dataclass
class CLIPerConfig:
    clip: CLIPTransformerConfig
    whisper: WhisperConfig
    temporal_head: TransformerHeadConfig
    out_features: int = 4
    clip_proj_dropout: float = 0.1
    whisper_proj_dropout: float = 0.1

    def __post_init__(self):
        if isinstance(self.clip, dict):
            self.clip = CLIPTransformerConfig(**self.clip)
        if isinstance(self.whisper, dict):
            self.whisper = WhisperConfig(**self.whisper)
        if isinstance(self.temporal_head, dict):
            self.temporal_head = TransformerHeadConfig(**self.temporal_head)

# CLIP
# class LayerNorm(nn.LayerNorm):
#     """Subclass torch's LayerNorm to handle fp16."""

#     def forward(self, x: torch.Tensor):
#         orig_type = x.dtype
#         ret = super().forward(x.type(torch.float32))
#         return ret.type(orig_type)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # For convenience
        width = config.width
        patch_size = config.patch_size
        input_resolution = config.image_resolution
        output_dim = config.output_dim

        self.input_resolution = input_resolution
        self.output_dim = config.output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        transformer_config = TransformerConfig(
            width=width,
            layers=config.layers,
            heads=config.heads,
            attn_dropout=0.0,
            mlp_dropout=0.0,
            attn_mask=None)
        self.transformer = Transformer(transformer_config)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, config.output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

# Whisper
class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x, weight, bias
    ):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x,
        xa = None,
        mask = None,
        kv_cache = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q, k, v, mask = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x,
        xa = None,
        mask = None,
        kv_cache = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self, n_mels, n_ctx, n_state, n_head, n_layer
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

# CLIPER
class CLIPer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.clip, clip_dim = self._init_clip()
        self.whisper, whisper_dim = self._init_whisper()

        # Features projections
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Dropout(config.clip.proj_dropout),
            nn.Linear(clip_dim, config.clip.transformer_head.width))
        self.clip_head = TransformerHead(config.clip.transformer_head)

        self.whisper_proj = nn.Sequential(
            nn.LayerNorm(whisper_dim),
            nn.Dropout(config.whisper.proj_dropout),
            nn.Linear(whisper_dim, config.whisper.transformer_head.width))
        self.whisper_head = TransformerHead(config.whisper.transformer_head)

        # Frames projections
        self.clip_frames_proj = nn.Sequential(
            nn.LayerNorm(config.clip.transformer_head.frames),
            nn.Dropout(config.clip_proj_dropout),
            nn.Linear(
                config.clip.transformer_head.frames, 
                config.temporal_head.frames))

        self.whisper_frames_proj = nn.Sequential(
            nn.LayerNorm(config.whisper.transformer_head.frames),
            nn.Dropout(config.whisper_proj_dropout),
            nn.Linear(
                config.whisper.transformer_head.frames, 
                config.temporal_head.frames))

        # Skip connection
        self.clip_out = nn.Linear(
            config.clip.transformer_head.out_features,
            config.temporal_head.out_features)

        self.whisper_out = nn.Linear(
            config.whisper.transformer_head.out_features,
            config.temporal_head.out_features)

        # Temporal head
        self.temporal_head = TransformerHead(config.temporal_head)

        self.temporal_out = nn.Linear(
            3 * config.temporal_head.out_features,
            config.out_features)

    def forward_clip(self, x):
        b = x.size(0)
        x = rearrange(x, 'b f c h w -> (b f) c h w') # (BxF, C, H, W)

        x = self.clip(x) # (BxF, D)
        x = x / x.norm(dim=-1, keepdim=True) # L2 normalize

        x = self.clip_proj(x) # (BxF, W)
        x = rearrange(x, '(b f) w -> b f w', b=b)

        x = self.clip_head(x)
        x = rearrange(x, 'b f w -> b w f')
        x = self.clip_frames_proj(x)
        x = rearrange(x, 'b w f -> b f w')
        return x

    def forward_whisper(self, y):
        y = self.whisper(y)
        y = self.whisper_proj(y)
        y = self.whisper_head(y)

        y = rearrange(y, 'b f w -> b w f')
        y = self.whisper_frames_proj(y)
        y = rearrange(y, 'b w f -> b f w')
        return y

    def forward(self, x, y):
        x = self.forward_clip(x)
        y = self.forward_whisper(y)

        xo = x.mean(dim=1)
        xo = self.clip_out(xo)

        yo = y.mean(dim=1)
        yo = self.whisper_out(yo)

        # x = x / x.norm(dim=-1, keepdim=True)
        x = nn.functional.normalize(x, p=2, dim=-1)
        # y = y / y.norm(dim=-1, keepdim=True)
        y = nn.functional.normalize(y, p=2, dim=-1)

        z = torch.cat([x, y], dim=-1)
        z = self.temporal_head(z)

        zo = torch.cat([xo, z, yo], dim=-1)
        zo = self.temporal_out(zo)

        return zo

    def _init_clip(self):
        clip_config = self.config.clip
        if clip_config.pretrained_weights is None:
            raise ValueError('Pretrained weights must be provided for CLIP')
        
        state_dict = torch.load(
            clip_config.pretrained_weights, map_location=self.device)

        # Get each parameter
        width = state_dict["conv1.weight"].shape[0]
        heads = width // 64
        layers = len([k for k in state_dict.keys() if k.endswith(".attn.in_proj_weight")])
        patch_size = state_dict["conv1.weight"].shape[-1]
        grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = patch_size * grid_size
        output_dim = state_dict["proj"].shape[1]

        config = VisionTransformerConfig(
            width=width,
            heads=heads,
            layers=layers,
            patch_size=patch_size,
            grid_size=grid_size,
            image_resolution=image_resolution,
            output_dim=output_dim)
        logger.info('VisionTransformer config:')
        logger.info(json.dumps(asdict(config), indent=4))

        clip = VisionTransformer(config)

        clip.load_state_dict(state_dict)
        logging.info(
            f'Loaded pretrained weights from {clip_config.pretrained_weights}')

        if clip_config.freeze:
            for param in clip.parameters():
                param.requires_grad = False
            logger.info('Frozen CLIP')

        return clip, output_dim

    def _init_whisper(self):
        whisper_config = self.config.whisper

        state_dict = torch.load(
            whisper_config.pretrained_weights,
            map_location=self.device)

        # Get each parameter from state_dict
        n_mels = state_dict["conv1.weight"].shape[1]
        n_state = state_dict["conv2.weight"].shape[0]
        n_head = n_state // 64
        n_ctx = state_dict["positional_embedding"].shape[0]
        n_layer = len(
            [k for k in state_dict.keys() if k.endswith('attn.query.weight')])

        config = {
            'n_mels': n_mels,
            'n_ctx': n_ctx,
            'n_state': n_state,
            'n_head': n_head,
            'n_layer': n_layer
        }
        logger.info('whisper encoder config:')
        logger.info(json.dumps(config, indent=4))

        whisper_encoder = WhisperAudioEncoder(**config)

        whisper_encoder.load_state_dict(state_dict)
        logger.info(
            f'Loaded pretrained weights from {whisper_config.pretrained_weights}')

        if whisper_config.freeze:
            for param in whisper_encoder.parameters():
                param.requires_grad = False
            logger.info('Frozen whisper')

        return whisper_encoder, n_state

def main():
    from torchinfo import summary
    logger.setLevel(logging.INFO)

    with open('configs/cliper_e.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = CLIPerConfig(**config['model'])

    cliper = CLIPer(config)
    x = torch.randn(2, 16, 3, 224, 224)
    y = torch.randn(2, 128, 3000)
    z = cliper(x, y)

    print(f'{z.shape=}')

    summary(cliper, input_size=[(2, 16, 3, 224, 224), (2, 128, 3000)])

    for name, param in cliper.named_parameters():
        if param.requires_grad:
            logger.info(f'{name=}, {param.grad=}')

if __name__ == '__main__':
    main()