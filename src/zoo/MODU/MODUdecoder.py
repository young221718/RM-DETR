from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.utils.denoising import get_contrastive_denoising_training_group
from src.utils.utils import bias_init_with_prob
from src.nn.transformer import (
    DinoTransformerDecoder,
    DeformableTransformerDecoderLayer,
    MLP,
)

from src.core import register


__all__ = ["MODUdecoder"]


@register
class MODUdecoder(nn.Module):
    __share__ = ["num_classes"]

    def __init__(
        self,
        num_classes=80,
        dim=256,
        num_queries=300,
        position_embed_type="sine",
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        is_decoder_pos=False,
        attn_swap=False,
    ):

        super().__init__()
        assert position_embed_type in [
            "sine",
            "learned",
        ], f"ValueError: position_embed_type not supported {position_embed_type}!"
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.dim = dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(
            dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
            is_decoder_pos=is_decoder_pos,
            attn_swap=attn_swap,
        )
        self.decoder = DinoTransformerDecoder(
            dim, decoder_layer, num_decoder_layers, eval_idx
        )

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, dim, padding_idx=num_classes
            )

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, dim)
        self.query_pos_head = MLP(4, 2 * dim, dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(
                dim,
            ),
        )
        self.enc_score_head = nn.Linear(dim, num_classes)
        self.enc_bbox_head = MLP(dim, dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(dim, num_classes) for _ in range(num_decoder_layers)]
        )
        self.dec_bbox_head = nn.ModuleList(
            [MLP(dim, dim, 4, num_layers=3) for _ in range(num_decoder_layers)]
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(in_channels, self.dim, 1, bias=False),
                            ),
                            (
                                "norm",
                                nn.BatchNorm2d(
                                    self.dim,
                                ),
                            ),
                        ]
                    )
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channels,
                                    self.dim,
                                    3,
                                    2,
                                    padding=1,
                                    bias=False,
                                ),
                            ),
                            ("norm", nn.BatchNorm2d(self.dim)),
                        ]
                    )
                )
            )
            in_channels = self.dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [
            0,
        ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None
    ):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(
                memory.device
            )

        # memory = torch.where(valid_mask, memory, 0)
        memory = (
            valid_mask.to(memory.dtype) * memory
        )  # TODO fix type error for onnx export

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(
            enc_outputs_class.max(-1).values, self.num_queries, dim=1
        )

        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(
                1, 1, enc_outputs_coord_unact.shape[-1]
            ),
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1
            )

        enc_topk_logits = enc_outputs_class.gather(
            dim=1,
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]),
        )

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(
                dim=1,
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]),
            )
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return (
            target,
            reference_points_unact.detach(),
            enc_topk_bboxes,
            enc_topk_logits,
            topk_ind,
        )

    def forward(self, feats, targets=None):
        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                )
            )
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits, topk_idx = (
            self._get_decoder_input(
                memory, spatial_shapes, denoising_class, denoising_bbox_unact
            )
        )

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out["aux_outputs"].extend(
                self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes])
            )

            if self.training and dn_meta is not None:
                out["dn_aux_outputs"] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class, outputs_coord)
        ]
