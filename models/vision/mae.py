import sys
import logging
import json
import yaml
from functools import partial
from typing import Callable
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
from einops import rearrange, repeat
import timm.models.vision_transformer as vit
from models.utils import TransformerHeadConfig, TransformerHead

logger = logging.getLogger(__name__)

@dataclass
class MAETransformerConfig:
    pretrained_weights: str
    transformer_head: TransformerHeadConfig
    proj_dropout: float = 0.1
    freeze: bool = True

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)


class MAETransformer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.mae, dim = self._init_mae()

        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(config.proj_dropout),
            nn.Linear(dim, config.transformer_head.width)
        )

        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        # x: B,F,C,H,W
        b = x.size(0)
        x = rearrange(x, 'b f c h w -> (b f) c h w') # (BxF, C, H, W)

        x = self.mae(x) # (BxF, D)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        x = self.proj(x) # (BxF, W)
        x = rearrange(x, '(b f) w -> b f w', b=b)

        x = self.transformer(x)
        return x

    def _init_mae(self):
        checkpoint = torch.load(
            self.config.pretrained_weights,
            map_location=self.device)
        # state_dict = checkpoint['model']
        # NOTE: Temporary fix to support old checkpoints
        logger.warning('Should fix the way pretrained weights are loaded')
        if self.config.pretrained_weights.endswith('mae_pretrain_vit_base.pth'):
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint['state_dict']
            state_dict = {
                k.replace('model.', ''): v for k, v in state_dict.items()
            }
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith('decoder_')
            }

        config = {
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_rate': 0.,
            'global_pool': 'avg',
            'norm_layer_eps': 1e-6,
            'num_classes': 0,
            'patch_size': 16,
            'attn_drop_rate': 0.1,
            'drop_path_rate': 0.1
        }
        logger.info('MAE config:')
        logger.info(json.dumps(config, indent=4))

        norm_layer_eps = config.pop('norm_layer_eps')
        model = vit.VisionTransformer(
            **config, 
            norm_layer=partial(nn.LayerNorm, eps=norm_layer_eps),
            fc_norm=nn.Identity())

        msg = model.load_state_dict(state_dict, strict=False)

        # breakpoint()

        if msg.missing_keys != ['mae.fc_norm.weight', 'mae.fc_norm.bias']:
            logger.warning(f'Missing keys: {msg.missing_keys}')

        model.fc_norm = nn.Identity()

        if self.config.freeze:
            for param in model.parameters():
                param.requires_grad = False
            logger.info('Frozen MAE')

        return model, config['embed_dim']

def main():
    from torchinfo import summary

    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/mae.yaml'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = MAETransformerConfig(**config['model'])

    model = MAETransformer(config)
    x = torch.randn(2, 16, 3, 224, 224)
    y = model(x)
    print(y.shape)

    summary(model, input_size=(2, 16, 3, 224, 224))

if __name__ == '__main__':
    main()