import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core import register


__all__ = [
    "MODU",
]


@register
class MODU(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        encoder_out = self.encoder(x)
        output = self.decoder(encoder_out, targets)

        return output

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
