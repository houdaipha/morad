import yaml
import logging
from dataclasses import dataclass

import torch
import transformers
import torch.nn as nn

@dataclass
class BertConfig:
    model_name: str = 'UBC-NLP/MARBERT'
    dropout: float = 0.1
    out_features: int = 4

class Bert(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device

        self.bert, hidden_size = self._init_bert()
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(hidden_size, config.out_features)

    def _init_bert(self):
        bert = transformers.AutoModel.from_pretrained(self.config.model_name)
        hidden_size = bert.embeddings.word_embeddings.weight.size(1)
        return bert, hidden_size

    def forward(self, x, mask=None):
        out = self.bert(x, attention_mask=mask)
        x = out[1] # classificatoin token + l
        x = self.dropout(x)
        x = self.out(x)
        return x

def main():
    from torchinfo import summary

    logging.basicConfig(level=logging.INFO)
    config_path = '/home/houdaifa.atou/main/code/morad/configs/text/bert.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = BertConfig(**config['model'])
    model = Bert(config)

    x = torch.randint(0, 100, (8, 32))
    mask = torch.randint(0, 2, (8, 32))
    y = model(x, mask)

    logging.info(f'Input: {x.size()}')
    logging.info(f'Output: {y.size()}')

    summary(model, input_size=(8, 32), dtypes=[torch.int64])

if __name__ == '__main__':
    main()