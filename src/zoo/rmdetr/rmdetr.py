import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.nn.transformer import (
    MLP,
)
from src.core import register


__all__ = [
    "RMDETR",
]


@register
class RMDETR(nn.Module):
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
        backbone_init_dim=64,
        mask_dim=256,
        query_dim=256,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale
        
        self.mask_embed = MLP(query_dim, query_dim*4, mask_dim, num_layers=3)
        self.mask_norm = nn.LayerNorm(query_dim)
        self.mask_features = nn.Conv2d(backbone_init_dim, mask_dim, kernel_size=1)

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        encoder_out = self.encoder(x[1:])
        output, out_query = self.decoder(encoder_out, targets)

        m1 = self.mask_features(x[0])
        m2 = F.interpolate(encoder_out[0], scale_factor=2, mode="bilinear", align_corners=False)
        mask = m1 + m2
        mask = self.get_masks(out_query[-1], mask)
        output["pred_masks"] = mask
        
        return output

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
    
    def get_masks(self, query, mask_features):
        mask_embed = self.mask_embed(self.mask_norm(query))
        masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # masks = F.sigmoid(masks)
        return masks
