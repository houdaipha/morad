import logging
import yaml
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

logger = logging.getLogger(__name__)

@dataclass
class VideoSwinConfig:
    out_features: int = 4
    pretrained_weights: str = 'kinetics400_imagenet22k' # KINETICS400_V1
    dropout: float = 0.

class VideoSwin(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        if config.pretrained_weights == 'kinetics400_imagenet22k':
            weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        elif config.pretrained_weights == 'kinetics400':
            weights = Swin3D_B_Weights.KINETICS400_V1
        else:
            weights = None

        self.swin = swin3d_b(weights=weights, dropout=config.dropout)

        in_features = self.swin.head.in_features
        if config.out_features != self.swin.num_classes:
            self.swin.head = nn.Linear(in_features, config.out_features)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W
        x = self.swin(x)
        return x


def main():
    from torchinfo import summary
    logging.basicConfig(level=logging.INFO)

    with open('configs/videoswin.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = VideoSwinConfig(**config['model'])
    model = VideoSwin(config)

    x = torch.randn(2, 16, 3, 224, 224)
    y = model(x)

    print(y.shape)
    print(summary(model, input_size=(2, 16, 3, 224, 224)))

if __name__ == '__main__':
    main()
