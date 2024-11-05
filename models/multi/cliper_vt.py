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
import transformers
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
class BertConfig:
    transformer_head: TransformerHeadConfig
    model_name: str = 'UBC-NLP/MARBERTv2'
    freeze: bool = False
    proj_dropout: float = 0.1

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)


@dataclass
class CLIPerConfig:
    clip: CLIPTransformerConfig
    bert: BertConfig
    temporal_head: TransformerHeadConfig
    out_features: int = 4
    clip_proj_dropout: float = 0.1
    bert_proj_dropout: float = 0.1

    def __post_init__(self):
        if isinstance(self.clip, dict):
            self.clip = CLIPTransformerConfig(**self.clip)
        if isinstance(self.bert, dict):
            self.bert = BertConfig(**self.bert)
        if isinstance(self.temporal_head, dict):
            self.temporal_head = TransformerHeadConfig(
                **self.temporal_head)

# CLIP
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

# CLIPER
class CLIPer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.clip, clip_dim = self._init_clip()
        self.bert, bert_dim = self._init_bert()

        # Features projections
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Dropout(config.clip.proj_dropout),
            nn.Linear(clip_dim, config.clip.transformer_head.width))
        self.clip_head = TransformerHead(config.clip.transformer_head)


        self.bert_proj = nn.Sequential(
            nn.LayerNorm(bert_dim),
            nn.Dropout(config.bert.proj_dropout),
            nn.Linear(bert_dim, config.bert.transformer_head.width))
        self.bert_head = TransformerHead(config.bert.transformer_head)


        # Frames projections
        self.clip_frames_proj = nn.Sequential(
            nn.LayerNorm(config.clip.transformer_head.frames),
            nn.Dropout(config.clip_proj_dropout),
            nn.Linear(
                config.clip.transformer_head.frames, 
                config.temporal_head.frames))

        self.bert_frames_proj = nn.Sequential(
            nn.LayerNorm(config.bert.transformer_head.frames),
            nn.Dropout(config.bert_proj_dropout),
            nn.Linear(
                config.bert.transformer_head.frames,
                config.temporal_head.frames))

        # Skip connection
        self.clip_out = nn.Linear(
            config.clip.transformer_head.out_features,
            config.temporal_head.out_features)

        self.bert_out = nn.Linear(
            config.bert.transformer_head.out_features,
            config.temporal_head.out_features)

        # Temporal head
        self.temporal_head_av = TransformerHead(config.temporal_head)
        self.temporal_head_at = TransformerHead(config.temporal_head)

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

    def forward_bert(self, z, mask=None):
        out = self.bert(z, mask)
        z = out.last_hidden_state
        z = self.bert_proj(z)
        z = self.bert_head(z)

        z = rearrange(z, 'b f w -> b w f')
        z = self.bert_frames_proj(z)
        z = rearrange(z, 'b w f -> b f w')
        return z

    def forward(self, x, z, mask=None):
        x = self.forward_clip(x)
        z = self.forward_bert(z, mask)

        xo = x.mean(dim=1)
        xo = self.clip_out(xo)

        zo = z.mean(dim=1)
        zo = self.bert_out(zo)

        # x = x / x.norm(dim=-1, keepdim=True)
        x = nn.functional.normalize(x, p=2, dim=-1)
        z = nn.functional.normalize(z, p=2, dim=-1)

        xz = torch.cat([x, z], dim=-1)
        xz = self.temporal_head_av(xz)


        zo = torch.cat([xo, xz, zo], dim=-1)
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

    def _init_bert(self):
        bert_config = self.config.bert
        bert = transformers.BertModel.from_pretrained(bert_config.model_name)
        hidden_size = bert.embeddings.word_embeddings.weight.size(1)

        if bert_config.freeze:
            for param in bert.parameters():
                param.requires_grad = False
            logger.info('Frozen Bert')

        # TODO: Add option to load model from saved weights
        return bert, hidden_size

def main():
    from torchinfo import summary
    logger.setLevel(logging.INFO)

    with open('configs/multi/cliper_vt.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = CLIPerConfig(**config['model'])

    cliper = CLIPer(config)
    x = torch.randn(2, 16, 3, 224, 224)
    z = torch.randint(0, 100, (2, 32))
    mask = torch.randint(0, 2, (2, 32))
    o = cliper(x, z, mask)

    print(f'{o.shape=}')

    summary(cliper, input_data=[x, z])

    for name, param in cliper.named_parameters():
        if param.requires_grad:
            logger.info(f'{name=}, {param.grad=}')

if __name__ == '__main__':
    main()