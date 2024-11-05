from training.base import TrainConfig
from training.base import BasePipeline as Pipeline

def forward(model, data, device):
    """Forward pass for visual"""
    frames = data['frames'].to(device)
    return model(frames)