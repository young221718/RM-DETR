import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.conv import MSCAN_CCFF
from src.nn.transformer import TransformerEncoder, TransformerEncoderLayer
from src.utils.utils import LayerNorm

from src.core import register


__all__ = ["MODUencoder"]


@register
class MODUencoder(nn.Module):
    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.dim = dim
        self.use_enc_idx = use_encoder_idx
        self.num_enc_layers = num_encoder_layers
        self.pe_temp = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, dim, kernel_size=1, bias=False),
                    nn.GELU(),
                    LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1),
                    nn.GELU(),
                    LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                )
            )
            self.fpn_blocks.append(
                MSCAN_CCFF(
                    dim * 2,
                    dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dim, dim, kernel_size=2, stride=2),
                )
            )
            self.pan_blocks.append(
                MSCAN_CCFF(
                    dim * 2,
                    dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                )
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_enc_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.dim,
                    self.pe_temp,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """ """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_enc_layers > 0:
            for i, enc_ind in enumerate(self.use_enc_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.dim, self.pe_temp
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(
                        src_flatten.device
                    )

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.dim, h, w).contiguous()
                )
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                torch.concat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        return outs
