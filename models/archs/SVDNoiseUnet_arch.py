import torch
import torch.nn as nn
import einops

from torch.nn import functional as F
from torch.jit import Final
from timm.layers import use_fused_attn

__all__ = ['SVDNoiseUnet', 'SVDNoiseUnet_Concise']

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TensorAdder(nn.Module):
    def __init__(self, in_ch1: int, in_ch2: int, out_ch: int = 256):
        super().__init__()
        # 1×1 convs to remap channels
        self.conv1 = nn.Conv2d(in_ch1, out_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(in_ch2, out_ch, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1: (N, in_ch1, H1, W1) e.g. (32, 1, 1, 192)
        x2: (N, in_ch2, H2, W2) e.g. (32, 1, 192, 192)
        returns (N, out_ch, H2, W2) e.g. (32, 256, 192, 192)
        """
        y1 = self.conv1(x1)    # → (N, 256, H1, W1)
        y2 = self.conv2(x2)    # → (N, 256, H2, W2)

        # if spatial dims differ, broadcast y1 to y2’s H×W
        if y1.shape[2:] != y2.shape[2:]:
            y1 = y1.expand(-1, -1, y2.size(2), y2.size(3))

        return y1 + y2
