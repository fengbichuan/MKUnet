import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile
import torchvision
import os
from functools import partial

import timm
from timm.layers import trunc_normal_tf_

from timm.models import named_apply

from einops import rearrange  # <--- 1. 已添加 einops 导入


# ===================================================================
# ===== 1. 你的 CosinConv2D 模块 (保持不变) ========================
# ===================================================================

class CosinConv2D(nn.Conv2d):
    """
    Cosine-Similarity Convolution (标量幂缩放)
    - 思路：先对 kernel 与局部 patch 做 L2 归一化，计算 cos 相似度；再做带符号的幂缩放 sign(x)*(|x|+eps)^p。
    - 关键点：
      1) 可选 shared weights（Depthwise 情况自动关闭）；
      2) p 为可学习参数并下限截断到 p_min；
      3) q 为可学习的正标量（通过 log_q 参数化），加入到输入范数分母，稳定训练。
    Inputs : x ∈ (B, C_in, H, W)
    Outputs: y ∈ (B, C_out, H_out, W_out)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 0,
            stride: int = 1,
            groups: int = 1,
            shared_weights: bool = False,
            w_max: float = 1.0,
            p_min: float = 0.1,
            q_init: float = 1e-3,
            eps: float = 1e-6,
    ):
        # 分组合法性
        assert groups == 1 or groups == in_channels, (
            "'groups' needs to be 1 or in_channels "
            f"({in_channels})."
        )
        assert out_channels % groups == 0, (
            f"out_channels ({out_channels}) must be a multiple of groups ({groups})."
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        # depthwise 情况禁用 shared_weights
        self.shared_weights = False if groups == in_channels else shared_weights  # <-- MODIFIED: 修正了你的逻辑 (之前是 groups == 1)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            padding=padding,
            stride=stride,
            groups=groups,
        )

        # 将 kernel_size 存为 int（假设方核）
        self.kernel_size = int(kernel_size)

        # 核心超参/形状
        self.channels_per_kernel = self.in_channels // self.groups
        self.w_max = float(w_max)
        self.p_min = float(p_min)
        self.eps = float(eps)

        # 需要学习的 kernel（支持 shared_weights）
        if self.shared_weights:
            self.n_kernels = self.out_channels // self.groups
        else:
            self.n_kernels = self.out_channels

        scaled_weight = np.random.uniform(
            low=-self.w_max, high=self.w_max,
            size=(self.n_kernels, self.channels_per_kernel, self.kernel_size, self.kernel_size)
        )
        # 覆盖父类的 weight
        self.weight = nn.Parameter(torch.as_tensor(scaled_weight, dtype=self.weight.dtype))

        # 学习的 p（逐输出通道/核）
        p_values = np.random.uniform(low=1.0, high=3.0, size=(1, self.n_kernels, 1, 1))
        self.p = nn.Parameter(torch.as_tensor(p_values, dtype=self.weight.dtype))

        # 学习的 q（正数，通过 log 参数化）
        self.log_q = nn.Parameter(torch.full((1, 1, 1, 1), float(np.log(q_init)), dtype=self.weight.dtype))

        # “全 1 卷积核”作为 buffer，用于输入范数卷积（按设备/类型在 forward 中转换）
        ones = torch.ones(
            self.groups, self.channels_per_kernel, self.kernel_size, self.kernel_size,
            dtype=self.weight.dtype
        )
        self.register_buffer("ones_kernel", ones, persistent=False)

        # ------------------ 核心计算 (已修正) ------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约束 learnable 参数范围
        # [FIX] 使用 torch.clamp (out-of-place) 代替 .clamp_() (inplace)
        # 不要在 forward 过程中 inplace 修改模型参数，这会导致 autograd 错误
        p_clamped = torch.clamp(self.p, min=self.p_min)
        weight_clamped = torch.clamp(self.weight, min=-self.w_max, max=self.w_max)

        q = torch.exp(self.log_q)

        # 根据 shared_weights 展开权重/参数
        if self.shared_weights:
            weight = weight_clamped.repeat(self.groups, 1, 1, 1)  # (C_out, Cin/G, k, k)
            p = p_clamped.repeat(1, self.groups, 1, 1)  # (1, C_out, 1, 1)
        else:
            weight = weight_clamped
            p = p_clamped

        return self._cosine_power_conv(x, weight, p, q)

    def _cosine_power_conv(self, x: torch.Tensor, weight: torch.Tensor, p: torch.Tensor,
                           q: torch.Tensor) -> torch.Tensor:
        # 1) 归一化 kernel
        w_norm = self._weight_norm(weight)  # (C_out,1,1,1)
        weight_n = weight / (w_norm + self.eps)

        # 2) 计算归一化后的 cos 相似度（分母为输入局部 L2 范数 + q）
        x_norm = self._input_norm(x, q)  # (B, C_out, H_out, W_out)
        cos_sim = F.conv2d(
            x, weight_n, stride=self.stride, padding=self.padding, groups=self.groups
        ) / (x_norm + self.eps)

        # 3) 带符号幂缩放
        return cos_sim.sign() * (cos_sim.abs() + self.eps) ** p

    def _weight_norm(self, weight: torch.Tensor) -> torch.Tensor:
        # 每个 kernel 的 L2 范数
        return weight.square().sum(dim=(1, 2, 3), keepdim=True).sqrt()

    def _input_norm(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # 用“全 1 卷积核”对 x^2 做卷积，相当于局部窗口 L2 范数（未开方）
        ones = self.ones_kernel.to(device=x.device, dtype=x.dtype)
        x2_sum = F.conv2d(
            x.square(),
            ones,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )  # (B, groups, H_out, W_out)

        # L2 范数 + q
        xnorm = (x2_sum + self.eps).sqrt() + q.to(device=x.device, dtype=x.dtype)

        # 按组扩展到 C_out 通道
        outputs_per_group = self.out_channels // self.groups
        return torch.repeat_interleave(xnorm, repeats=outputs_per_group, dim=1)


# ===================================================================
# ===== 2. 你的 MK_UNet 模型（继续）=================================
# ===================================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):  # <-- CosinConv2D 也会被初始化
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            # CosinConv2D 没有 kernel_size 属性，但它有 self.kernel_size (int)
            if hasattr(module, 'kernel_size') and isinstance(module.kernel_size, tuple):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            elif hasattr(module, 'kernel_size') and isinstance(module.kernel_size, int):
                fan_out = module.kernel_size * module.kernel_size * module.out_channels
            else:
                # 假设为 3x3，作为后备
                fan_out = 9 * module.out_channels

            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    # CosinConv2D 的 p 和 log_q 参数在 __init__ 中有自己的初始化，这里跳过


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes == None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activation = act_layer(activation, inplace=True)

        # 1x1 卷积，保持为 nn.Conv2d
        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2

        # <-- 已替换为 CosinConv2D -->
        self.conv = CosinConv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            groups=1
        )
        # self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) # 原始代码

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class GroupedAttentionGate(nn.Module):
    # (这个模块现在未被 MK_UNet 使用，但保留定义)
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GroupedAttentionGate, self).__init__()
        if kernel_size == 1:
            groups = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# ===================================================================
# ===== 2B. 粘贴新的 LCA 模块 (HVI 论文) =============================
# ===================================================================

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class NormDownsample(nn.Module):
    # (这个模块未被 MK_UNet 使用，但保留定义)
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))

    def forward(self, x):
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class NormUpsample(nn.Module):
    # (这个模块未被 MK_UNet 使用，但保留定义)
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


class CAB(nn.Module):
    # (Cross Attention Block - LCA 的依赖)
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer (LCA 的依赖)
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention (LCA)
class HV_LCA(nn.Module):
    # (这是 LCA 的变体 1)
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    # (这是 LCA 的变体 2，我们将使用这个)
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x


# ===================================================================
# ===== 3. 你的 MK_UNet 模型（继续）================================
# ===================================================================

class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                # <-- 已替换为 CosinConv2D -->
                CosinConv2D(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=self.in_channels
                ),
                # nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, # 原始代码
                #           groups=self.in_channels, bias=False),                                      # 原始代码
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)

        # 1x1 逐点卷积，保持为 nn.Conv2d
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        # 这里调用了 MultiKernelDepthwiseConv，已在上面修改
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales

        # 1x1 逐点卷积，保持为 nn.Conv2d
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            # 1x1 逐点卷积，保持为 nn.Conv2d
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)

        # <-- 修正channel_shuffle的组 -->
        current_channels = dout.shape[1]
        dout = channel_shuffle(dout, gcd(current_channels, self.out_c))

        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                      activation='relu6'):
    """
    create a series of multi-kernel inverted residual blocks.
    """
    convs = []
    xx = MultiKernelInvertedResidualBlock(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel,
                                          add=add, kernel_sizes=kernel_sizes, activation=activation)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MultiKernelInvertedResidualBlock(out_c, out_c, 1, expansion_factor=expansion_factor,
                                                  dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes,
                                                  activation=activation)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv


class MK_UNet(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2,
                 lca_heads=8,  # <--- 1. MODIFIED: 替换了 gag_kernel
                 **kwargs):
        super().__init__()

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        # --- 2. MODIFIED: 已用 I_LCA 替换 GroupedAttentionGate ---
        self.LCA1 = I_LCA(dim=channels[3], num_heads=lca_heads, bias=False)
        self.LCA2 = I_LCA(dim=channels[2], num_heads=lca_heads, bias=False)
        self.LCA3 = I_LCA(dim=channels[1], num_heads=lca_heads, bias=False)
        self.LCA4 = I_LCA(dim=channels[0], num_heads=lca_heads, bias=False)
        # -----------------------------------------------------------

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor,
                                          dw_parallel=True, add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        # 1x1 输出卷积，保持为 nn.Conv2d
        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out

        ### Stage 4
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        ### Bottleneck
        out = F.max_pool2d(self.encoder5(out), 2, 2)

        ### Stage 4
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))

        # --- 3. MODIFIED: 更新了 Attention Gate 调用 ---
        # 原始: t4 = self.AG1(g=out, x=t4)
        t4 = self.LCA1(x=t4, y=out)  # (x=Query from Encoder, y=Context from Decoder)
        out = torch.add(out, t4)

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')

        # --- 4. MODIFIED: 更新了 Attention Gate 调用 ---
        # 原始: t3 = self.AG2(g=out, x=t3)
        t3 = self.LCA2(x=t3, y=out)
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')

        # --- 5. MODIFIED: 更新了 Attention Gate 调用 ---
        # 原始: t2 = self.AG3(g=out, x=t2)
        t2 = self.LCA3(x=t2, y=out)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')

        # --- 6. MODIFIED: 更新了 Attention Gate 调用 ---
        # 原始: t1 = self.AG4(g=out, x=t1)
        t1 = self.LCA4(x=t1, y=out)
        out = torch.add(out, t1)

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return [p4]  # [p4, p3, p2, p1]


# EOF
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        # --- 7. MODIFIED: 更新了测试代码 ---
        input = torch.randn(1, 3, 256, 256).cuda()

        # 使用的参数 (lca_heads=8)
        model = MK_UNet(num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                        kernel_sizes=[1, 3, 5], expansion_factor=2,
                        lca_heads=8).to(torch.device('cuda:0'))  # <-- 已更新参数

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with CosinConv2D and LCA) ---")  # <-- 已更新打印信息
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output[0].shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")
        print("You also need to install 'einops': pip install einops")