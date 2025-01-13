import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import math

from src.utils.utils import deformable_attention_core_func
from src.core import register


@register
class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
    ):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim,
            self.total_points * 2,
        )
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(
            1, 1, -1, 1
        )
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self, query, reference_points, value, value_spatial_shapes, value_mask=None
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        return output


@register
class MSDeformableAttention_condition(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        style:str='a'
    ):
        """
        Multi-Scale Deformable Attention Module
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points * 2

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(
            embed_dim*2,
            self.total_points*2
        )
        
        self.attention_weights = nn.Linear(embed_dim*2, self.total_points)

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters(style)

    def generate_square_points(self, n_heads, n_points):
        points = []
        side_length = n_points*2
        half_side = side_length / 2
        step = side_length / (n_heads // 4)
        
        for i in range(n_heads // 4):
            points.append((i * step - half_side, half_side))  # Top side
        for i in range(n_heads // 4):
            points.append((half_side, half_side - i * step))  # Right side
        for i in range(n_heads // 4):
            points.append((half_side - i * step, -half_side))  # Bottom side
        for i in range(n_heads // 4):
            points.append((-half_side, -half_side + i * step))  # Left side
        
        return points
    
    def _reset_parameters(self, style):
        
        if style == 'a':
            print('style a')
            init.constant_(self.sampling_offsets.weight, 0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            content_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            content_init = content_init / content_init.abs().max(-1, keepdim=True).values
            content_init = content_init.reshape(self.num_heads, 1, 1, 2).tile(
                [1, self.num_levels, self.num_points, 1]
            )
            scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(
                1, 1, -1, 1
            )
            content_init *= scaling

            position_init = []
            outline = torch.tensor(self.generate_square_points(self.num_heads,self.num_points)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)

            for i, p in enumerate(outline):
                position_init.append(p + t * (outline[(i + 1) % len(outline)] - p))
                # print(out_init[-1])
            position_init = torch.stack(position_init)
            position_init = position_init.reshape(self.num_heads, 1, self.num_points, 2).tile(
                [1, self.num_levels, 1, 1]
            )

            init_bias = torch.cat([content_init, position_init], dim=0)
            self.sampling_offsets.bias.data[...] = init_bias.flatten()
        elif style == 'b':
            print('style b')
            init.constant_(self.sampling_offsets.weight, 0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            content_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            content_init = content_init / content_init.abs().max(-1, keepdim=True).values
            content_init = content_init.reshape(self.num_heads, 1, 1, 2).tile(
                [1, self.num_levels, self.num_points, 1]
            )
            scaling = torch.linspace(1, self.num_points-0.5, self.num_points).reshape(1, 1, -1, 1)
            content_init *= scaling

            position_init = []
            outline = torch.tensor(self.generate_square_points(self.num_heads,self.num_points)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)

            for i, p in enumerate(outline):
                position_init.append(p + t * (outline[(i + 1) % len(outline)] - p))
                # print(out_init[-1])
            position_init = torch.stack(position_init)
            position_init = position_init.reshape(self.num_heads, 1, self.num_points, 2).tile(
                [1, self.num_levels, 1, 1]
            )

            init_bias = torch.cat([content_init, position_init], dim=0)
            self.sampling_offsets.bias.data[...] = init_bias.flatten()
        elif style == 'c':
            print('style c')
            init.constant_(self.sampling_offsets.weight, 0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            content_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            content_init = content_init / content_init.abs().max(-1, keepdim=True).values
            content_init = content_init.reshape(self.num_heads, 1, 1, 2).tile(
                [1, self.num_levels, self.num_points, 1]
            )
            scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(
                    1, 1, -1, 1
                )
            content_init *= scaling

            position_init = []
            outline_inner = torch.tensor(self.generate_square_points(self.num_heads // 2, self.num_points)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)
            for i, p in enumerate(outline_inner):
                position_init.append(p + t * (outline_inner[(i + 1) % len(outline_inner)] - p))

            outline_outter = torch.tensor(self.generate_square_points(self.num_heads // 2, self.num_points+0.5)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)
            for i, p in enumerate(outline_outter):
                position_init.append(p + t * (outline_outter[(i + 1) % len(outline_outter)] - p))
            position_init = torch.stack(position_init)
            position_init = position_init.reshape(self.num_heads, 1, self.num_points, 2).tile(
                [1, self.num_levels, 1, 1]
            )

            init_bias = torch.cat([content_init, position_init], dim=0)
            self.sampling_offsets.bias.data[...] = init_bias.flatten()
        elif style == 'd':
            print('style d')
            init.constant_(self.sampling_offsets.weight, 0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            content_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            content_init = content_init / content_init.abs().max(-1, keepdim=True).values
            content_init = content_init.reshape(self.num_heads, 1, 1, 2).tile(
                [1, self.num_levels, self.num_points, 1]
            )
            scaling = torch.linspace(1, self.num_points-0.5, self.num_points).reshape(1, 1, -1, 1)
            content_init *= scaling

            position_init = []
            outline_inner = torch.tensor(self.generate_square_points(self.num_heads // 2, self.num_points)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)
            for i, p in enumerate(outline_inner):
                position_init.append(p + t * (outline_inner[(i + 1) % len(outline_inner)] - p))

            outline_outter = torch.tensor(self.generate_square_points(self.num_heads // 2, self.num_points+0.5)).reshape(-1, 2)
            t = torch.linspace(0, 1, self.num_points).view(-1, 1)
            for i, p in enumerate(outline_outter):
                position_init.append(p + t * (outline_outter[(i + 1) % len(outline_outter)] - p))
            position_init = torch.stack(position_init)
            position_init = position_init.reshape(self.num_heads, 1, self.num_points, 2).tile(
                [1, self.num_levels, 1, 1]
            )

            init_bias = torch.cat([content_init, position_init], dim=0)
            self.sampling_offsets.bias.data[...] = init_bias.flatten()
        
        
        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(
        self, query, reference_points, value, value_spatial_shapes, value_mask=None
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads * 2, -1)

        # content_sampling_offsets = self.sampling_content_offsets(
        #     query[..., : self.embed_dim]
        # ).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        # position_sampling_offsets = self.sampling_position_offsets(
        #     query[..., self.embed_dim :]
        # ).reshape(bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        # sampling_offsets = torch.cat(
        #     [content_sampling_offsets, position_sampling_offsets], dim=2
        # )
        # print(query.shape)
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads*2, self.num_levels, self.num_points, 2
        )

   
        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads*2, self.num_levels * self.num_points)
        # print(attention_weights.shape)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads * 2, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        return output


