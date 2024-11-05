import sys
import logging
import json
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
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

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

# From CLIP: github.com/openai/CLIP/
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

class CLIPTransformer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.clip, embed_dim = self._init_clip()

        if config.freeze:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(config.proj_dropout),
            nn.Linear(embed_dim, config.transformer_head.width))
        
        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        # x: B,F,C,H,W
        b = x.size(0)
        x = rearrange(x, 'b f c h w -> (b f) c h w') # (BxF, C, H, W)

        x = self.clip(x) # (BxF, D)
        x = x / x.norm(dim=-1, keepdim=True) # L2 normalize

        x = self.proj(x) # (BxF, W)
        x = rearrange(x, '(b f) w -> b f w', b=b)

        x = self.transformer(x)
        return x



    def _init_clip(self):
        if self.config.pretrained_weights is None:
            raise ValueError('Pretrained weights must be provided for CLIP')
        
        state_dict = torch.load(
            self.config.pretrained_weights, map_location=self.device)

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
            f'Loaded pretrained weights from {self.config.pretrained_weights}')
        return clip, output_dim

# Write a main fucntion to test CLIPTransformer
def main():
    from torchinfo import summary

    config = CLIPConfig(
        transformer_head=TransformerHeadConfig(
            frames=16,
            width=512,
            layers=3,
            heads=8,
            attn_dropout=0.0,
            mlp_dropout=0.0),
        pretrained_weights='/morad_dir/modelsWeights/clip_vit_b_16.pt',
        proj_dropout=0.1,
        freeze=True)

    model = CLIPTransformer(config)
    x = torch.randn(2, 16, 3, 224, 224)
    y = model(x)
    print(y.shape)

    summary(model, input_size=(2, 16, 3, 224, 224))

if __name__ == '__main__':
    main()
