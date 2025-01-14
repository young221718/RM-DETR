import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform

from .attention import *
from src.utils.utils import get_activation, inverse_sigmoid


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        deformable_attn: nn.Module = MSDeformableAttention,
        is_decoder_pos: bool = False,
        attn_swap=False,
    ):
        super().__init__()
        self.attn_swap = attn_swap
        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        if deformable_attn is None:
            self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        else:
            self.cross_attn = deformable_attn(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.is_decoder_pos = is_decoder_pos
        # self._reset_parameters()

    # def _reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             xavier_uniform(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        reference_points,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        attn_mask=None,
        memory_mask=None,
        query_pos_embed=None,
    ):

        if not self.attn_swap:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos_embed)
            tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # cross attention
            query = (
                self.with_pos_embed(tgt, query_pos_embed)
                if not self.is_decoder_pos
                else torch.cat([tgt, query_pos_embed], dim=-1)
            )

            tgt2 = self.cross_attn(
                query,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_mask,
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        else:
            # cross attention
            query = (
                self.with_pos_embed(tgt, query_pos_embed)
                if not self.is_decoder_pos
                else torch.cat([tgt, query_pos_embed], dim=-1)
            )

            tgt2 = self.cross_attn(
                query,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_mask,
            )
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # self attention
            q = k = self.with_pos_embed(tgt, query_pos_embed)
            tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DinoTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = (
                inter_ref_bbox.detach() if self.training else inter_ref_bbox
            )

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class MaskDinoTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        memory_level_start_index,
        bbox_head,
        score_head,
        query_pos_head,
        attn_mask=None,
        memory_mask=None,
    ):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_queries = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed,
            )

            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach)
            )

            if self.training:
                dec_out_logits.append(score_head[i](output))
                dec_out_queries.append(output)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                    )

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_queries.append(output)
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = (
                inter_ref_bbox.detach() if self.training else inter_ref_bbox
            )

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_queries),
        )
