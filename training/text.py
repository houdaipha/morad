from training.base import TrainConfig
from training.base import BasePipeline as Pipeline

def forward(model, data, device):
    """Forward pass for textual"""
    tokens = data['tokens'].to(device)
    mask = data['attention_mask'].to(device) if 'attention_mask' in data else None
    return model(tokens, mask)