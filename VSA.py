import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DecayPos1d(nn.Module):
    """
    1D 衰减相对位置先验（按 head 设定不同衰减速率）
    - 输入：序列长度 L
    - 输出：形状 (num_heads, L, L) 的相对位置衰减矩阵
    """

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float, heads_range: float):
        super().__init__()
        # 频率角频率（未直接用到，保留以兼容可能的扩展）
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        # 每个 head 一个衰减速率（越靠后 head 衰减越慢/快，取决于 heads_range）
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int) -> torch.Tensor:
        idx = torch.arange(l, device=self.decay.device)
        dist = (idx[:, None] - idx[None, :]).abs()              # (L, L)
        mask = dist * self.decay[:, None, None]                 # (H, L, L)
        return mask

    def forward(self, slen: int) -> torch.Tensor:
        return self.generate_1d_decay(int(slen))


class VolSelfAttention(nn.Module):
    """
    Volumetric Self-Attention
    - Token 维度自注意力（带窗口相对位置偏置）
    - 频谱/通道重排分支（Conv1x1 + DWConv）
    - 简单体素式融合（用空间注意力图加权）
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))         # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)                           # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                   # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        # token 自注意力（线性投影）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 频谱位置先验：按 head 提供 (H, C_per_head, C_per_head) 的衰减偏置
        self.realPos = DecayPos1d(embed_dim=64, num_heads=num_heads, initial_value=2, heads_range=4)

        # 频谱分支：Conv1x1 + 深度可分离卷积
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # 简单空间注意力门控（把 (B,H,N,N) 池化成 (B,N,1) 作为加权）
        self.Gao_spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_heads, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, C)，其中 N=Wh*Ww 必须与 window_size 匹配
        return: (B, N, C)
        """
        B, N, C = x.shape
        Wh, Ww = self.window_size
        assert N == Wh * Ww, f"N ({N}) 必须等于 window_size={self.window_size} 的乘积"
        hh = int(math.isqrt(N))
        assert hh * hh == N, "N 应为完全平方数，确保可重排为 (hh, hh)"

        # -------- Token 维自注意力 --------
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                            # (B, H, N, C//H)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                            # (B, H, N, N)

        # 相对位置偏置
        rel_pos = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1) # (N,N,H)
        rel_pos = rel_pos.permute(2, 0, 1).contiguous()                                                 # (H,N,N)
        attn = attn + rel_pos.unsqueeze(0)                                                              # (B,H,N,N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj_drop(self.proj(x1))

        # -------- 频谱/通道重排分支 --------
        # 频谱先验（按每头通道数）
        c_per_head = C // self.num_heads
        realPos = self.realPos(c_per_head)                                     # (H, CpH, CpH)

        x_s = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)                # (B,C,hh,hh)
        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)
        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos   # 广播到 (B,H,CpH,CpH)
        attn_c = attn_c.softmax(dim=-1)

        x2 = (attn_c @ v_c)                                                   # (B,H,CpH,(hh*hh))
        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=hh, w=hh)
        x2 = self.proj_C(x2)                                                  # (B,C,hh,hh)
        x2 = rearrange(x2, 'b c h w -> b (h w) c', h=hh, w=hh)                # (B,N,C)

        # -------- 体素式融合（空间权重）--------
        # 将 (B,H,N,N) → (B,64,1,1) → reshape 到 (B,N,1) 作为加权门控
        attn_spatial = self.Gao_spatial_attention(attn)                       # (B,64,1,1) 这里 64=示例中固定输出通道
        Bsa, _, _, _ = attn_spatial.shape
        attn_spatial = attn_spatial.reshape(Bsa, N, 1)
        x4 = attn_spatial * x2

        out = x1 + x2 + x4
        return out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# ==================== 示例（统一“五步模板”）====================
if __name__ == "__main__":
    # 1) 配置
    B, H, W, C = 2, 8, 8, 64              # N=H*W=64，需与 window_size 匹配
    N = H * W
    heads = 8
    win = (H, W)

    # 2) 构造输入：形状 (B, N, C)
    x = torch.randn(B, N, C)

    # 3) 实例化模块（统一命名 block）
    block = VolSelfAttention(dim=C, window_size=win, num_heads=heads, qkv_bias=True)

    # 4) 前向计算（可选：no_grad 测速/检形状）
    with torch.no_grad():
        y = block(x)

    # 5) 打印结构与张量形状
    print(block)
    print("x.shape =", x.shape)   # [B, N, C]
    print("y.shape =", y.shape)   # [B, N, C]
