from training.base import TrainConfig
from training.base import BasePipeline as Pipeline

def forward_av(model, data, device):
    """Forward pass for audio-visual"""
    frames = data['frames'].to(device)
    audio = data['audio'].to(device)
    return model(frames, audio)
    
def forward_vt(model, data, device):
    """Forward pass for visual-text"""
    frames = data['frames'].to(device)
    tokens = data['tokens'].to(device)
    mask = data['attention_mask'].to(device) if 'attention_mask' in data else None
    return model(frames, tokens, mask)

def forward_avt(model, data, device):
    """Forward pass for audio-visual-text"""
    frames = data['frames'].to(device)
    audio = data['audio'].to(device)
    tokens = data['tokens'].to(device)
    mask = data['attention_mask'].to(device) if 'attention_mask' in data else None
    return model(frames, audio, tokens, mask)


def forward_at(model, data, device):
    """Forward pass for audio-visual-text"""
    audio = data['audio'].to(device)
    tokens = data['tokens'].to(device)
    mask = data['attention_mask'].to(device) if 'attention_mask' in data else None
    return model(audio, tokens, mask)