import yaml
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MfLstmConfig:
    input_size: int = 13
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    head_dropout: float = 0.1
    head_hidden_dim: Optional[int] = None
    out_features: int = 4

    def __post_init__(self):
        if self.head_hidden_dim is None:
            self.head_hidden_dim = self.hidden_size // 2


class MfLstm(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )

        if config.bidirectional:
            lstm_out_dim = config.hidden_size * 2
        else:
            lstm_out_dim = config.hidden_size
        
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, config.out_features)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.head(x[:, -1, :])
        return x

def main():
    with open('configs/audio/mflstm.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = MfLstmConfig(**config['model'])
    model = MfLstm(config)

    x = torch.randn(2, 16, 13)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    main()