import os
import logging
import json
import yaml

from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.utils import TransformerHeadConfig, TransformerHead

logger = logging.getLogger(__name__)

@dataclass
class ResnetTransformerConfig:
    pretrained_weights: str
    transformer_head: TransformerHeadConfig
    freeze: bool = True
    proj_dropout: float = 0.1

    def __post_init__(self):
        if isinstance(self.transformer_head, dict):
            self.transformer_head = TransformerHeadConfig(
                **self.transformer_head)

# NOTE: Source https://github.com/openai/CLIP
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the
        # second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the
            # subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads: int,
            output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
            self,
            layers,
            output_dim,
            heads,
            input_resolution=224,
            width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width // 2,
            width,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class ResnetTransformer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device

        self.resnet, output_dim = self._init_resnet()

        self.proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(config.proj_dropout),
            nn.Linear(output_dim, config.transformer_head.width)
        )

        self.transformer = TransformerHead(config.transformer_head)

    def forward(self, x):
        b = x.size(0)
        x = rearrange(x, 'b f c h w -> (b f) c h w') # (BxF, C, H, W)

        x = self.resnet(x)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        x = self.proj(x) # (BxF, W)
        x = rearrange(x, '(b f) w -> b f w', b=b)

        x = self.transformer(x)
        return x

    def _init_resnet(self):
        state_dict = torch.load(
            self.config.pretrained_weights,
            map_location=self.device)

        counts: list = [
            len(set(k.split(".")[1] for k in state_dict if k.startswith(f"layer{b}"))) 
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
        output_dim = state_dict["attnpool.c_proj.weight"].shape[0]

        config = {
            'layers': vision_layers,
            'output_dim': output_dim,
            'heads': vision_width * 32 // 64,
            'input_resolution': image_resolution,
            'width': vision_width
        }
        logger.info('ResNet config:')
        logger.info(json.dumps(config, indent=4))

        resnet = ModifiedResNet(**config)
        resnet.load_state_dict(state_dict)

        if self.config.freeze:
            for param in resnet.parameters():
                param.requires_grad = False
            logger.info('ResNet is frozen')

        return resnet, output_dim


def main():
    from torchinfo import summary

    logger.setLevel(logging.INFO)

    with open('/home/houdaifa.atou/main/code/morad/configs/resnet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config = ResnetTransformerConfig(**config['model'])
    model = ResnetTransformer(config)

    x = torch.randn(2, 16, 3, 224, 224)
    y = model(x)
    print(y.shape)

    summary(model, input_size=(2, 16, 3, 224, 224))


if __name__ == '__main__':
    main()
