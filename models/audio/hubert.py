import sys
import math
import logging
import yaml
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model
from models.utils import TransformerHeadConfig, TransformerHead

logger = logging.getLogger(__name__)

HUBERT_CONFIG = {
    'extractor_mode': 'layer_norm',
    'extractor_conv_layer_config': [
        [512, 10, 5],
        [512, 3, 2],
        [512, 3, 2],
        [512, 3, 2],
        [512, 3, 2],
        [512, 2, 2],
        [512, 2, 2]],
    'extractor_conv_bias': True,
    'encoder_embed_dim': 1024,
    'encoder_projection_dropout': 0.0,
    'encoder_pos_conv_kernel': 128,
    'encoder_pos_conv_groups': 16,
    'encoder_num_layers': 24,
    'encoder_num_heads': 16,
    'encoder_attention_dropout': 0.1,
    'encoder_ff_interm_features': 4096,
    'encoder_ff_interm_dropout': 0.1,
    'encoder_dropout': 0.1,
    'encoder_layer_norm_first': True,
    'encoder_layer_drop': 0.1
}

@dataclass
class HubertConfig:
    transformer_head: TransformerHeadConfig
    model: dict = field(default_factory=lambda: HUBERT_CONFIG)
    freeze: bool = True
    proj_dropout: float = 0.1
    pretrained_weights: str = None
    # hubert_out_features: int = field(init=False)

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)
        # self.hubert_out_features = self.transformer_head.width

class Hubert(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.hubert, width = self._init_hubert()
        self.width = width

        self.proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(config.proj_dropout),
            nn.Linear(width, config.transformer_head.width)
        )

        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        x, _ = self.hubert(x)
        x = x / x.norm(dim=-1, keepdim=True) # Normalize

        x = self.proj(x)
        x = self.transformer(x)
        return x
        

    def _init_hubert(self):
        # Logg hubert config used
        hubert_config = self.config.model

        # Loading hubert model
        hubert = wav2vec2_model(**hubert_config, aux_num_out=None)
        if self.config.pretrained_weights is not None:
            state_dict = torch.load(
                self.config.pretrained_weights,
                map_location=self.device)
            hubert.load_state_dict(state_dict)
            logger.info(
                f'Loaded pretrained weights from {self.config.pretrained_weights}')

        # Freeze hubert
        if self.config.freeze:
            for p in hubert.parameters():
                p.requires_grad = False
            logger.info('Freezed Hubert model.')

        # Change output features
        dim = hubert_config['encoder_embed_dim']
        # if self.config.hubert_out_features != dim:
        #     # hubert.aux = nn.Linear(dim, self.config.hubert_out_features)
        #     hubert.aux = nn.Sequential(
        #         nn.Linear(dim, dim),
        #         nn.GELU(),
        #         nn.Dropout(0.2),
        #         nn.Linear(dim, self.config.hubert_out_features)
        #     )
        #     for p in hubert.aux.parameters():
        #         p.requires_grad = True
        #     logger.info("Hubet output features changed.")

        return hubert, dim

def main():
    from torchinfo import summary

    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/hubert.yaml'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = HubertConfig(**config['model'])
    model = Hubert(config)

    x = torch.randn(8, 80640)
    y = model(x)
    
    logger.info(y.shape)

    summary(model, (8, 80640))

if __name__ == "__main__":
    main()