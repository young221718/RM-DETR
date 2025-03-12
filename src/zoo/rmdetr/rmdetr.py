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
        # self.mask_norm = nn.LayerNorm(query_dim)
        self.m1 = nn.Sequential(
            nn.Conv2d(backbone_init_dim, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
        )
        self.m2 = nn.Sequential(
            nn.Conv2d(backbone_init_dim, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
        )
        self.fconv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
            nn.ReLU(),
        )
        
       
        
        nn.Conv2d(backbone_init_dim, mask_dim, kernel_size=1)

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        encoder_out = self.encoder(x[1:])
        output, out_query = self.decoder(encoder_out, targets)

        mask = self.get_mask(out_query[-1], output["pred_boxes"], encoder_out[0], x[0])
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
    
    def get_mask(self, query, box, from_encoder, from_backbone):
        m1 = self.m1(from_backbone)
        m2 = F.interpolate(from_encoder, scale_factor=2, mode="bilinear", align_corners=False)
        m2 = self.m2(m2)
        mask = m1 + m2
        mask = self.fconv(mask)
        n_query = box.shape[1]
        mask_embed = self.mask_embed(query[:,-n_query:,:])
        mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask)
        
        return mask
    
        