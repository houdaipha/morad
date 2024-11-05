from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import TimesformerModel

@dataclass
class TimesformerConfig:
    out_features: int = 4
    name: str = 'facebook/timesformer-base-finetuned-k400'

class Timesformer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config

        self.vivit = TimesformerModel.from_pretrained(config.name)

        hidden_size = self.vivit.layernorm.normalized_shape[0]
        
        self.classifier = nn.Linear(hidden_size, config.out_features)

    def forward(self, x):
        outputs = self.vivit(x)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])
        return logits

def main():
    config = TimesformerConfig(out_features=4)
    model = Timesformer(config)
    
    x = torch.randn(2, 16, 3, 224, 224)
    y = model(x)

    print(y.shape)

if __name__ == '__main__':
    main()
