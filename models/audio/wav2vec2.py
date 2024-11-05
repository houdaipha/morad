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

WAV2VEC_CONFIG = {
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
    'encoder_embed_dim': 1920,
    'encoder_projection_dropout': 0.0,
    'encoder_pos_conv_kernel': 128,
    'encoder_pos_conv_groups': 16,
    'encoder_num_layers': 48,
    'encoder_num_heads': 16,
    'encoder_attention_dropout': 0.1,
    'encoder_ff_interm_features': 7680,
    'encoder_ff_interm_dropout': 0.0,
    'encoder_dropout': 0.1,
    'encoder_layer_norm_first': True,
    'encoder_layer_drop': 0.1
}

@dataclass
class Wav2vec2Config:
    transformer_head: TransformerHeadConfig
    model: dict = field(default_factory=lambda: WAV2VEC_CONFIG)
    freeze: bool = True
    proj_dropout: float = 0.1
    pretrained_weights: str = None

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)

class Wav2vec2(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.wav2vec, width = self._init_wav2vec()
        self.width = width

        self.proj = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(config.proj_dropout),
            nn.Linear(width, config.transformer_head.width)
        )

        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        x, _ = self.wav2vec(x)
        x = x / x.norm(dim=-1, keepdim=True) # Normalize

        x = self.proj(x)
        x = self.transformer(x)
        return x
        

    def _init_wav2vec(self):
        # Logg wav2vec config used
        wav2vec_config = self.config.model

        # Loading wav2vec model
        wav2vec = wav2vec2_model(**wav2vec_config, aux_num_out=None)
        if self.config.pretrained_weights is not None:
            state_dict = torch.load(
                self.config.pretrained_weights,
                map_location=self.device)
            wav2vec.load_state_dict(state_dict)
            logger.info(
                f'Loaded pretrained weights from {self.config.pretrained_weights}')

        # Freeze feature extractor
        for p in wav2vec.feature_extractor.parameters():
            p.requires_grad = False
        logger.info('Freezed feature extractor.')

        # Freeze wav2vec
        if self.config.freeze:
            for p in wav2vec.parameters():
                p.requires_grad = False
            logger.info('Freezed wav2vec model.')

        # Change output features
        dim = wav2vec_config['encoder_embed_dim']

        return wav2vec, dim

def main():
    from torchinfo import summary

    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/wav2vec.yaml'

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = Wav2vec2Config(**config['model'])
    model = Wav2vec2(config)

    x = torch.randn(8, 80640)
    y = model(x)
    
    logger.info(y.shape)

    summary(model, (8, 80640))

if __name__ == "__main__":
    main()