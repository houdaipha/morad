from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import VivitModel

@dataclass
class VivitConfig:
    out_features: int = 4
    name: str = 'google/vivit-b-16x2-kinetics400'

class Vivit(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config

        self.vivit = VivitModel.from_pretrained(
            config.name,
            add_pooling_layer=False)

        hidden_size = self.vivit.layernorm.normalized_shape[0]
        
        self.classifier = nn.Linear(hidden_size, config.out_features)

    def forward(self, x):
        outputs = self.vivit(x)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])
        return logits

def main():
    config = VivitConfig(out_features=4)
    model = Vivit(config)
    
    x = torch.randn(2, 32, 3, 224, 224)
    y = model(x)

    print(y.shape)

if __name__ == '__main__':
    main()
