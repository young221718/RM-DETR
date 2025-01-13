import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.utils import get_activation, LayerNorm, GRN


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        padding=None,
        bias=False,
        act=None,
        norm="bn",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        if norm == "bn":
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm == "ln":
            self.norm = LayerNorm(ch_out, data_format="channels_first")
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return "{num_features}, eps={eps}".format(**self.__dict__)


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
    ):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.conv2 = ConvNormLayer(
            in_channels, hidden_channels, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[
                RepVggBlock(hidden_channels, hidden_channels, act=act)
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(
                hidden_channels, out_channels, 1, 1, bias=bias, act=act
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# ===============================================================
# 최찬영의 코딩 작성 구간
# ===============================================================


class MSCAN_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )

        # MSCA
        self.dconv5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.dconv7_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.dconv7_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.dconv11_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.dconv11_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.dconv21_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.dconv21_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv_mix = nn.Conv2d(dim, dim, 1)
        # MSCA end

        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        x = self.conv_in(x)

        # MSCA
        attn = self.dconv5(x)

        attn_0 = self.dconv7_1(attn)
        attn_0 = self.dconv7_2(attn_0)

        attn_1 = self.dconv11_1(attn)
        attn_1 = self.dconv11_2(attn_1)

        attn_2 = self.dconv21_1(attn)
        attn_2 = self.dconv21_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv_mix(attn)
        x = x * attn
        # MSCA end

        x = self.conv_out(x)
        return x


class MSCAN_Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.norm1 = LayerNorm(d_model, data_format="channels_first")
        self.attn = MSCAN_Attention(d_model)

        self.norm2 = LayerNorm(d_model, data_format="channels_first")
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1),
        )

    def forward(self, x):

        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x = x + x1

        x1 = self.norm2(x)
        x1 = self.ffn(x1)
        x = x + x1

        return x


class MSCAN_CCFF(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=None,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=bias),
            nn.GELU(),
            LayerNorm(hidden_channels, data_format="channels_first"),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=bias),
            nn.GELU(),
            LayerNorm(hidden_channels, data_format="channels_first"),
        )

        self.bottlenecks = nn.Sequential(
            *[MSCAN_Block(hidden_channels) for _ in range(num_blocks)]
        )
        if hidden_channels != out_channels:
            self.conv3 = nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, 1, bias=bias),
                nn.GELU(),
                LayerNorm(out_channels, data_format="channels_first"),
            )
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
