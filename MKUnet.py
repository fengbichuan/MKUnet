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


# ===================================================================
# ===== 1. 您的 CosinConv2D 模块 (无变动) ============================
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
        self.shared_weights = False if groups == in_channels else shared_weights

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约束 learnable 参数范围
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
# ===== 2. 引入的新模块 (SID-Lite 和 CDFA) ============================
# ===================================================================

# --- 2a. SID_Lite 模块 (来自 7、AMS（AAAI 2025）.py 的精简版) ---

class SID_CBR(nn.Module):
    """ SID-Lite 依赖的 CBR 模块 """

    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 标志是否应用激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 使用ReLU增加非线性

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class SID_Lite(nn.Module):
    """
    Semantic_Information_Decoupling (精简版)
    仅保留特征解耦分支，移除了大型上采样辅助分支以节省参数。
    """

    def __init__(self, in_c, out_c):
        super(SID_Lite, self).__init__()
        # 前景特征提取分支
        self.cbr_fg = nn.Sequential(
            SID_CBR(in_c, in_c // 2, kernel_size=3, padding=1),
            SID_CBR(in_c // 2, out_c, kernel_size=3, padding=1),
        )
        # 背景特征提取分支
        self.cbr_bg = nn.Sequential(
            SID_CBR(in_c, in_c // 2, kernel_size=3, padding=1),
            SID_CBR(in_c // 2, out_c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 提取前景、背景特征
        f_fg = self.cbr_fg(x)
        f_bg = self.cbr_bg(x)
        return f_fg, f_bg


# --- 2b. CDFA 模块 (来自 5、CDFA (AAAI 2025).py) ---

class CDFA_CBR(nn.Module):
    """ CDFA 依赖的 CBR 模块 """

    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ContrastDrivenFeatureAggregation(nn.Module):
    """
    来自 5、CDFA (AAAI 2025).py
    - x: 必须有 in_c 个通道
    - fg, bg: 必须有 out_c 个通道
    """

    def __init__(self, in_c, out_c, num_heads=4, kernel_size=3, padding=1, stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim = out_c  # 模块的 "工作维度"
        self.dim = dim
        self.in_c = in_c
        self.out_c = out_c

        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # 确保 head_dim 至少为 1
        self.head_dim = max(1, dim // num_heads)

        # (FIX for potential num_heads > dim) 确保 inner_dim 正确
        self.inner_dim = self.head_dim * self.num_heads

        self.scale = self.head_dim ** -0.5

        # 线性变换层，用于生成值特征
        self.v = nn.Linear(dim, self.inner_dim)  # v 映射到 inner_dim
        # 两个线性层分别生成前景和背景的注意力权重
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * self.num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(self.inner_dim, dim)  # 从 inner_dim 映射回 dim
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        # 预处理模块：将 in_c 映射到 dim (out_c)
        self.input_cbr = nn.Sequential(
            CDFA_CBR(in_c, dim, kernel_size=3, padding=1),
            CDFA_CBR(dim, dim, kernel_size=3, padding=1),
        )
        # 后处理模块：
        self.output_cbr = nn.Sequential(
            CDFA_CBR(dim, dim, kernel_size=3, padding=1),
            CDFA_CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        # x (B, in_c, H, W) -> (B, dim, H, W)
        x = self.input_cbr(x)

        # fg, bg 已经是 (B, dim, H, W)

        # 将输入转换为 (B, H, W, C) 格式
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        assert C == self.dim, f"Input x has {C} channels, but expected dim={self.dim}"

        v = self.v(x).permute(0, 3, 1, 2)  # (B, inner_dim, H, W)
        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)

        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)  # (B, H, W, inner_dim)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)  # (B, H, W, inner_dim)

        # Proj
        x_projected = self.proj(x_weighted_bg)
        x_projected = self.proj_drop(x_projected)  # (B, H, W, dim)
        x_projected = x_projected.permute(0, 3, 1, 2)  # (B, dim, H, W)

        out = self.output_cbr(x_projected)
        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # feature_map (fg/bg) 已经是 (B, H, W, C=dim)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        # attn (B, num_heads, L, K*K, K*K)
        # v (B, num_heads, L, K*K, head_dim)
        x_weighted = (attn @ v)  # (B, num_heads, L, K*K, head_dim)

        # (B, num_heads, L, K*K, head_dim) -> (B, num_heads, head_dim, K*K, L) -> (B, inner_dim * K*K, L)
        x_weighted = x_weighted.permute(0, 1, 4, 3, 2).reshape(B, self.inner_dim * self.kernel_size * self.kernel_size,
                                                               -1)

        # Fold
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding,
                            stride=self.stride)

        # (B, inner_dim, H, W) -> (B, H, W, inner_dim)
        x_weighted = x_weighted.permute(0, 2, 3, 1)
        return x_weighted


# ===================================================================
# ===== 3. 您的 MK_UNet 模型（已修改）=================================
# ===================================================================

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):  # CosinConv2D 也会被初始化
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
            if hasattr(module, 'kernel_size') and isinstance(module.kernel_size, tuple):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            elif hasattr(module, 'kernel_size') and isinstance(module.kernel_size, int):
                fan_out = module.kernel_size * module.kernel_size * module.out_channels
            else:
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
    """ 您的原始 ChannelAttention 模块 """

    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        # 确保 reduced_channels 至少为 1
        if self.in_planes < ratio:
            self.reduced_channels = 1
        else:
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
    """ 您的原始 SpatialAttention 模块 (已应用 CosinConv2D) """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2

        # <-- 使用 CosinConv2D -->
        self.conv = CosinConv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            groups=1
        )

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


# ----- (已移除 GroupedAttentionGate, 它将被 CDFA 替换) -----


class MultiKernelDepthwiseConv(nn.Module):
    """ 您的原始 MultiKernelDepthwiseConv 模块 (已恢复) """

    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                # <-- MODIFIED: 使用 CosinConv2D -->
                CosinConv2D(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=self.in_channels
                ),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    """ 您的原始 MultiKernelInvertedResidualBlock 模块 (已恢复) """

    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_c = int(self.in_c * expansion_factor)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
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
    """ 您的原始 mk_irb_bottleneck 模块 (已恢复) """
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

    # (MODIFIED: 恢复 kernel_sizes, 移除 gag_kernel)
    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5], expansion_factor=2, **kwargs):
        super().__init__()

        # --- 恢复原始 Encoder/Decoder ---
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

        # --- (MODIFIED) 添加 SID 模块 ---
        # SID_Lite 接收瓶颈层 C4=160, 输出 C4=160
        self.sid = SID_Lite(in_c=channels[4], out_c=channels[4])

        # --- (MODIFIED) 添加 CDFA 模块 替换 AG ---
        # CDFA(in_c, out_c) -> x: in_c, fg/bg: out_c
        self.cdfa1 = ContrastDrivenFeatureAggregation(in_c=channels[3], out_c=channels[3])  # 96, 96
        self.cdfa2 = ContrastDrivenFeatureAggregation(in_c=channels[2], out_c=channels[2])  # 64, 64
        self.cdfa3 = ContrastDrivenFeatureAggregation(in_c=channels[1], out_c=channels[1])  # 32, 32
        self.cdfa4 = ContrastDrivenFeatureAggregation(in_c=channels[0], out_c=channels[0])  # 16, 16

        # --- (MODIFIED) 添加 1x1 卷积用于投影 fg/bg
        self.fg_proj1 = nn.Conv2d(channels[4], channels[3], kernel_size=1)  # 160 -> 96
        self.bg_proj1 = nn.Conv2d(channels[4], channels[3], kernel_size=1)
        self.fg_proj2 = nn.Conv2d(channels[4], channels[2], kernel_size=1)  # 160 -> 64
        self.bg_proj2 = nn.Conv2d(channels[4], channels[2], kernel_size=1)
        self.fg_proj3 = nn.Conv2d(channels[4], channels[1], kernel_size=1)  # 160 -> 32
        self.bg_proj3 = nn.Conv2d(channels[4], channels[1], kernel_size=1)
        self.fg_proj4 = nn.Conv2d(channels[4], channels[0], kernel_size=1)  # 160 -> 16
        self.bg_proj4 = nn.Conv2d(channels[4], channels[0], kernel_size=1)

        # --- 原始 CA / SA / Out 保持不变 ---
        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], num_classes, kernel_size=1)  # 64
        self.out2 = nn.Conv2d(channels[1], num_classes, kernel_size=1)  # 32
        self.out3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)  # 16
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)  # 16

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        B = x.shape[0]
        ### Encoder
        ### Stage 1
        out = F.max_pool2d(self.encoder1(x), 2, 2)
        t1 = out  # B, C0, H/2, W/2
        ### Stage 2
        out = F.max_pool2d(self.encoder2(out), 2, 2)
        t2 = out  # B, C1, H/4, W/4
        ### Stage 3
        out = F.max_pool2d(self.encoder3(out), 2, 2)
        t3 = out  # B, C2, H/8, W/8
        ### Stage 4
        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out  # B, C3, H/16, W/16

        ### Bottleneck
        out = F.max_pool2d(self.encoder5(out), 2, 2)  # B, C4, H/32, W/32

        ### (MODIFIED) 提取语义特征
        f_fg, f_bg = self.sid(out)  # B, C4, H/32, W/32

        ### Stage 4
        out = self.CA1(out) * out
        out = self.SA(out) * out
        up_out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))  # B, C3, H/16, W/16

        # (MODIFIED) 使用 CDFA 替换 AG
        fg_guide_s4 = self.fg_proj1(F.interpolate(f_fg, scale_factor=2, mode='bilinear'))
        bg_guide_s4 = self.bg_proj1(F.interpolate(f_bg, scale_factor=2, mode='bilinear'))
        t4_fused = self.cdfa1(x=t4, fg=fg_guide_s4, bg=bg_guide_s4)  # 使用 CDFA 重构 t4
        out = torch.add(up_out, t4_fused)  # 融合

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        up_out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))  # B, C2, H/8, W/8

        # --- START FIX ---
        p1 = F.interpolate(self.out1(up_out), scale_factor=(8, 8), mode='bilinear')
        # --- END FIX ---

        # (MODIFIED)
        fg_guide_s3 = self.fg_proj2(F.interpolate(f_fg, scale_factor=4, mode='bilinear'))
        bg_guide_s3 = self.bg_proj2(F.interpolate(f_bg, scale_factor=4, mode='bilinear'))
        t3_fused = self.cdfa2(x=t3, fg=fg_guide_s3, bg=bg_guide_s3)
        out = torch.add(up_out, t3_fused)

        ### Stage 2
        out = self.CA3(out) * out
        out = self.SA(out) * out
        up_out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))  # B, C1, H/4, W/4

        # --- START FIX ---
        p2 = F.interpolate(self.out2(up_out), scale_factor=(4, 4), mode='bilinear')
        # --- END FIX ---

        # (MODIFIED)
        fg_guide_s2 = self.fg_proj3(F.interpolate(f_fg, scale_factor=8, mode='bilinear'))
        bg_guide_s2 = self.bg_proj3(F.interpolate(f_bg, scale_factor=8, mode='bilinear'))
        t2_fused = self.cdfa3(x=t2, fg=fg_guide_s2, bg=bg_guide_s2)
        out = torch.add(up_out, t2_fused)

        ### Stage 1
        out = self.CA4(out) * out
        out = self.SA(out) * out
        up_out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))  # B, C0, H/2, W/2

        # --- START FIX ---
        p3 = F.interpolate(self.out3(up_out), scale_factor=(2, 2), mode='bilinear')
        # --- END FIX ---

        # (MODIFIED)
        fg_guide_s1 = self.fg_proj4(F.interpolate(f_fg, scale_factor=16, mode='bilinear'))
        bg_guide_s1 = self.bg_proj4(F.interpolate(f_bg, scale_factor=16, mode='bilinear'))
        t1_fused = self.cdfa4(x=t1, fg=fg_guide_s1, bg=bg_guide_s1)
        out = torch.add(up_out, t1_fused)

        ### Final Stage
        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        p4 = self.out4(out)

        return [p4]  # [p4, p3, p2, p1]


# EOF
if __name__ == '__main__':
    # 确保有可用的CUDA设备
    if torch.cuda.is_available():
        input = torch.randn(1, 3, 256, 256).cuda()

        # (MODIFIED: 恢复原始参数)
        model = MK_UNet(num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                        kernel_sizes=[1, 3, 5], expansion_factor=2).to(torch.device('cuda:0'))

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with SID + CDFA) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output[0].shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")