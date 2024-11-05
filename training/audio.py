from training.base import TrainConfig
from training.base import BasePipeline as Pipeline

def forward(model, data, device):
    """Forward pass for audio"""
    audio = data['audio'].to(device)
    return model(audio)