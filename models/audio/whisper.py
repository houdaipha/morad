import logging
import json
import yaml
from collections import OrderedDict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import TransformerHeadConfig, TransformerHead

logger = logging.getLogger(__name__)


@dataclass
class WhisperConfig:
    pretrained_weights: str
    freeze: bool
    proj_dropout: float
    transformer_head: TransformerHeadConfig

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x, weight, bias
    ):
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x,
        xa = None,
        mask = None,
        kv_cache = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q, k, v, mask = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x,
        xa = None,
        mask = None,
        kv_cache = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self, n_mels, n_ctx, n_state, n_head, n_layer
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

# Whisper
class Whisper(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.whisper_encoder, n_state = self._init_whisper()

        self.proj = nn.Sequential(
            nn.LayerNorm(n_state),
            nn.Dropout(config.proj_dropout),
            nn.Linear(n_state, config.transformer_head.width))

        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        x = self.whisper_encoder(x) # B, 1500, 1280
        x = self.proj(x)
        x = self.transformer(x)
        return x

    def _init_whisper(self):
        state_dict = torch.load(
            self.config.pretrained_weights,
            map_location=self.device)

        # Get each parameter from state_dict
        n_mels = state_dict["conv1.weight"].shape[1]
        n_state = state_dict["conv2.weight"].shape[0]
        n_head = n_state // 64
        n_ctx = state_dict["positional_embedding"].shape[0]
        n_layer = len(
            [k for k in state_dict.keys() if k.endswith('attn.query.weight')])

        config = {
            'n_mels': n_mels,
            'n_ctx': n_ctx,
            'n_state': n_state,
            'n_head': n_head,
            'n_layer': n_layer
        }
        logger.info('whisper encoder config:')
        logger.info(json.dumps(config, indent=4))

        whisper_encoder = WhisperAudioEncoder(**config)

        whisper_encoder.load_state_dict(state_dict)
        logger.info(
            f'Loaded pretrained weights from {self.config.pretrained_weights}')

        if self.config.freeze:
            for param in whisper_encoder.parameters():
                param.requires_grad = False
            logger.info('Frozen whisper encoder')

        return whisper_encoder, n_state

def main():
    logging.basicConfig(level=logging.INFO)

    with open('configs/whisper.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = WhisperConfig(**config['model'])
    model = Whisper(config)

    x = torch.randn(2, 128, 3000)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    main()
