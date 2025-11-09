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
# ===== 2. 引入的新模块 (来自 8、PConv(AAAI2025).py) ==================
# ===================================================================

# CBR模块：卷积 -> 批归一化 -> 可选ReLU激活
class Dilated_CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act  # 控制是否激活
        # 构建卷积和归一化组合层
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)  # 使用ReLU激活函数
        # ai缝合大王

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


# 通道注意力模块：结合全局平均和最大池化，生成通道权重
class Dilated_channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Dilated_channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        # --- START FIX ---
        # 确保缩减后的通道数至少为 1
        reduced_channels = max(1, in_planes // ratio)
        # --- END FIX ---

        self.fc1 = nn.Conv2d(in_planes, reduced_channels, kernel_size=1, bias=False)  # 通道降维
        self.relu1 = nn.ReLU()  # 激活
        self.fc2 = nn.Conv2d(reduced_channels, in_planes, kernel_size=1, bias=False)  # 通道升维

        self.sigmoid = nn.Sigmoid()  # 映射到 [0, 1]

    def forward(self, x):
        x_orig = x  # 保留原始输入
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x_orig * self.sigmoid(out)


# 空间注意力模块：利用通道统计生成空间权重图
class Dilated_spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Dilated_spatial_attention, self).__init__()
        # 核尺寸必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # 将卷积输出映射至 [0,1]
        # ai缝合大王

    def forward(self, x):
        x_orig = x  # 保存原始特征
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对通道取平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对通道取最大值
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_conv = self.conv1(x_cat)
        return x_orig * self.sigmoid(x_conv)


# 膨胀卷积模块：采用不同膨胀率的卷积以捕捉多尺度上下文信息，并通过注意力机制进行特征重构
class Dilated_conv_Module(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 每个分支先用CBR模块做卷积，再结合通道注意力
        self.c1 = nn.Sequential(Dilated_CBR(in_c, out_c, kernel_size=1, padding=0), Dilated_channel_attention(out_c))
        self.c2 = nn.Sequential(Dilated_CBR(in_c, out_c, kernel_size=3, padding=6, dilation=6),
                                Dilated_channel_attention(out_c))
        self.c3 = nn.Sequential(Dilated_CBR(in_c, out_c, kernel_size=3, padding=12, dilation=12),
                                Dilated_channel_attention(out_c))
        self.c4 = nn.Sequential(Dilated_CBR(in_c, out_c, kernel_size=3, padding=18, dilation=18),
                                Dilated_channel_attention(out_c))
        # 合并分支后使用3x3卷积整合多尺度信息
        self.c5 = Dilated_CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        # 直接1x1卷积对输入进行映射
        self.c6 = Dilated_CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = Dilated_spatial_attention()  # 空间注意力模块

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        # 将四个分支输出在通道维度上拼接
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x_out = self.relu(xc + xs)  # 残差连接和激活
        x_out = self.sa(x_out)  # 空间注意力加权
        return x_out


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


class GroupedAttentionGate(nn.Module):
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


# ----- (已移除 MultiKernelDepthwiseConv) -----


class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    (MODIFIED: 使用 Dilated_conv_Module 替换 MultiKernelDepthwiseConv)
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=2, activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)

        # 1x1 逐点卷积
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        # <-- MODIFIED: 使用新的多尺度（膨胀）模块 -->
        self.multi_scale_dwconv = nn.Sequential(
            Dilated_conv_Module(self.ex_c, self.ex_c),
            nn.BatchNorm2d(self.ex_c),  # 添加 BN + Act 以匹配原始 DWConv 块
            act_layer(activation, inplace=True)
        )

        # 1x1 逐点卷积
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.ex_c, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )

        if self.use_skip_connection and (self.in_c != self.out_c):
            # 1x1 逐点卷积
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)

        # <-- MODIFIED: 简化了前向传播 -->
        dout = self.multi_scale_dwconv(pout1)

        current_channels = dout.shape[1]
        dout = channel_shuffle(dout, gcd(current_channels, self.out_c))

        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2, activation='relu6'):
    """
    create a series of multi-kernel inverted residual blocks.
    (MODIFIED: 移除了 kernel_sizes, add, dw_parallel 参数)
    """
    convs = []
    xx = MultiKernelInvertedResidualBlock(in_c, out_c, s, expansion_factor=expansion_factor,
                                          activation=activation)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MultiKernelInvertedResidualBlock(out_c, out_c, 1, expansion_factor=expansion_factor,
                                                  activation=activation)
            convs.append(xx)
    conv = nn.Sequential(*convs)
    return conv


class MK_UNet(nn.Module):

    # (MODIFIED: 移除了 kernel_sizes)
    def __init__(self, num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                 expansion_factor=2, gag_kernel=3, **kwargs):
        super().__init__()

        # (MODIFIED: 调用时移除了 kernel_sizes, add, dw_parallel)
        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1, expansion_factor=expansion_factor)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1, expansion_factor=expansion_factor)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1, expansion_factor=expansion_factor)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1, expansion_factor=expansion_factor)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1, expansion_factor=expansion_factor)

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=channels[3] // 2)
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=channels[2] // 2)
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=channels[1] // 2)
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=channels[0] // 2)

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1, expansion_factor=expansion_factor)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1, expansion_factor=expansion_factor)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1, expansion_factor=expansion_factor)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1, expansion_factor=expansion_factor)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1, expansion_factor=expansion_factor)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()

        # 1x1 输出卷积
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
        t4 = self.AG1(g=out, x=t4)
        out = torch.add(out, t4)

        ### Stage 3
        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        p1 = F.interpolate(self.out1(out), scale_factor=(8, 8), mode='bilinear')
        t3 = self.AG2(g=out, x=t3)
        out = torch.add(out, t3)

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        p2 = F.interpolate(self.out2(out), scale_factor=(4, 4), mode='bilinear')
        t2 = self.AG3(g=out, x=t2)
        out = torch.add(out, t2)

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        p3 = F.interpolate(self.out3(out), scale_factor=(2, 2), mode='bilinear')
        t1 = self.AG4(g=out, x=t1)
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
        input = torch.randn(1, 3, 256, 256).cuda()

        # (MODIFIED: 移除了 kernel_sizes)
        model = MK_UNet(num_classes=1, in_channels=3, channels=[16, 32, 64, 96, 160], depths=[1, 1, 1, 1, 1],
                        expansion_factor=2, gag_kernel=3).to(torch.device('cuda:0'))

        flops, params = profile(model, inputs=(input,))
        output = model(input)

        print(f"\n--- Final Model Output (with Dilated_conv_Module) ---")
        print(f"Input shape: {input.shape}")
        print(f"Output shape: {output[0].shape}")
        print(f"FLOPs (G): {flops / 1e9}")
        print(f"Params (M): {params / 1e6}")
    else:
        print("CUDA not available. Please run this on a machine with a GPU.")