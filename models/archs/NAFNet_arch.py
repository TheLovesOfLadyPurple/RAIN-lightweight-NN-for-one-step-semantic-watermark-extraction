# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.arch_util import LayerNorm2d
from models.archs.local_arch import Local_Base
from abc import abstractmethod
import math
from einops import repeat, rearrange
# from attention import LinearSpatialTransformer
from inspect import isfunction
import torch_dct as dct
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    # assert len(w.shape) == 1
    # w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(device=w.device)
    # Fix: Reshape emb to [2, 160] first, then multiply with w
    emb = emb.unsqueeze(0).expand(w.shape[0], -1)  # Shape becomes [2, 160]
    emb = w.to(dtype) * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
  
class CrossAttention(nn.Module): # wrong name
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.sg = SimpleGate()

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)

        if context.dim() == 4:
            if context.shape[-1] == 1:
                context = context.squeeze(-1)
            elif context.shape[-2] == 1:
                context = context.squeeze(-2)

        if context.dim() == 3 and context.shape[-1] != self.context_dim:
            if context.shape[1] == self.context_dim:
                context = context.transpose(1, 2)
            else:
                raise ValueError(f"Unexpected context shape {context.shape}, expected last dim {self.context_dim}")

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # linear与否与上面无关
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        
        # linear attention
        # q = q.permute(0,2,1)
        # k = k.permute(0,2,1)
        # k = self.sg(k) + 1.0
        # q = self.sg(q) + 1.0
        # k = k.permute(0,2,1)
        # q = q.permute(0,2,1)
        # sim = torch.einsum('b i j, b i d -> b j d', k, v) * self.scale
        # attn = sim
        # out = torch.einsum('b i j, b j d -> b i d', q, attn)
        
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], use_time_embed = False, prompt_channel_width=None):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.use_time_embed = use_time_embed

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        
        self.last_layer = list(self.decoders[-1].children())[-1]

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        # print(inp.size())
        # print(inp.shape)
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 4, 64, 64), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)



if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)