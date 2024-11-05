from functools import partial
from dataclasses import dataclass
import yaml

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights

RESNETS = {
    'resnet50': partial(resnet50, weights=ResNet50_Weights.DEFAULT),
    'resnet101': partial(resnet101, weights=ResNet101_Weights.DEFAULT),
}

@dataclass
class MsCNNConfig:
    resnet: str = 'resnet50'
    out_features: int = 4

class MsCNN(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config

        self.resnet = RESNETS[config.resnet]()

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, 
            config.out_features)

    def forward(self, x):
        x = self.resnet(x)
        return x

def main():
    with open('configs/audio/mscnn.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = MsCNNConfig(**config['model'])
    model = MsCNN(config)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print(y.shape)

if __name__ == '__main__':
    main()

