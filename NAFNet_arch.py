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
from archs.arch_util import LayerNorm2d
from archs.local_arch import Local_Base
from abc import abstractmethod
import math
from einops import repeat, rearrange
# from attention import LinearSpatialTransformer
from inspect import isfunction

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
  
class LightweightLinearCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.sg = SimpleGate()

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
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
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., head_nums=4):
        super().__init__()
        dw_channel = c * DW_Expand
        self.head_nums = head_nums
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        assert (dw_channel // 2) % self.head_nums == 0, "Channel count must be divisible by head_nums for SCA"
        per_head_channels = (dw_channel // 2) // self.head_nums
        self.sca = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=per_head_channels, out_channels=per_head_channels, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            ) for _ in range(self.head_nums)
        ])

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
        needs_reshape = inp.dim() == 3

        if needs_reshape:
            b, seq_len, c = inp.shape
            spatial = int(math.sqrt(seq_len))
            assert spatial * spatial == seq_len, f"Sequence length {seq_len} is not a perfect square"
            inp_4d = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)
        else:
            inp_4d = inp

        x = self.norm1(inp_4d)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        if self.head_nums > 1:
            head_chunks = torch.chunk(x, self.head_nums, dim=1)
            attended = [chunk * sca(chunk) for chunk, sca in zip(head_chunks, self.sca)]
            x = torch.cat(attended, dim=1)
        else:
            x = x * self.sca[0](x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp_4d + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        out = y + x * self.gamma

        if needs_reshape:
            out = rearrange(out, 'b c h w -> b (h w) c')

        return out


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], use_time_embed = False):
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


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimeEmbedLinearAttentionNAFBlock):
                x = layer(x, emb, context)
            elif isinstance(layer, LinearAttentionNAFBlock):
                x = layer(x, context)
            elif isinstance(layer, TimestepBlock) or isinstance(layer,TimeEmbedResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class TimeEmbedNAFBlock(TimestepBlock):
    def __init__(self, c,org_time_embed_dim = 320, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.org_time_embed_dim = org_time_embed_dim
        self.input_time_embed_dim = self.org_time_embed_dim * 4
        self.emb_layers = nn.Sequential(
            SimpleGate(),
            nn.Conv2d(in_channels=self.input_time_embed_dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        
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

    def forward(self, inp, time_emb):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        emb_out = self.emb_layers(time_emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x + emb_out
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


class TimeEmbedNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, org_time_embed_dim = 320, enc_blk_nums=[], dec_blk_nums=[] ,use_time_embed = True):
        super().__init__()
        
        self.use_time_embed = use_time_embed

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.org_time_embed_dim = org_time_embed_dim
        self.time_embed_dim = self.org_time_embed_dim * 4
        
        self.time_embed = nn.Sequential(
            nn.Conv2d(in_channels=self.org_time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True),
            SimpleGate(),
            nn.Conv2d(in_channels=self.time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True)
        )

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                TimestepEmbedSequential(
                    *[TimeEmbedNAFBlock(chan, org_time_embed_dim = self.org_time_embed_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            TimestepEmbedSequential(
                *[TimeEmbedNAFBlock(chan, org_time_embed_dim = self.org_time_embed_dim) for _ in range(middle_blk_num)]
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
                TimestepEmbedSequential(
                    *[TimeEmbedNAFBlock(chan, org_time_embed_dim = self.org_time_embed_dim) for _ in range(num)]
                )
            )
        self.last_layer = list(self.decoders[-1].children())[-1]

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, timesteps=torch.tensor([1000.0]), idx = None):
        # print(inp.size())
        # print(inp.shape)
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        timesteps = timesteps.flatten()

        encs = []
        # print('timesteps', timesteps)
        t_emb = timestep_embedding(timesteps, self.org_time_embed_dim, repeat_only=False)
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]
        idx_emb = guidance_scale_embedding(w=idx,embedding_dim=t_emb.shape[1]).to(device=x.device)
        
        # print('t_emb',t_emb.shape)
        emb = self.time_embed(t_emb.to(device=x.device))
        emb = emb + idx_emb

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x,emb)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x,emb)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x,emb)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class TimeEmbedResBlock(TimestepBlock):
    def __init__(self, c,org_time_embed_dim = 320, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        
        self.org_time_embed_dim = org_time_embed_dim
        self.input_time_embed_dim = self.org_time_embed_dim * 4
        self.emb_layers = nn.Sequential(
            SimpleGate(),
            nn.Conv2d(in_channels=self.input_time_embed_dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        # SimpleGate
        self.sg = SimpleGate()
        
        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, time_emb):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        emb_out = self.emb_layers(time_emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x + emb_out
        x = self.sg(x)
        x = self.dropout1(x)
        return inp + x * self.beta
    
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttentionTransBlock(TimestepBlock):
    def __init__(self, c, num_heads = 8):
        super().__init__()
        self.attention_group_norm = Normalize(c)
        self.cha = c
        self.num_heads = num_heads
        self.dim_head = c // self.num_heads
        inner_dim = self.num_heads * self.dim_head
        self.proj_in = nn.Conv2d(c,
                                 inner_dim, # I am not sure whether it is a good idea to expand dim in this place
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
    #     self.proj_in.register_full_backward_hook(self.custom_hook)
        
    # def custom_hook(module, grad_input, grad_output):
    #     # 钩子函数的实现
    #     print("Gradient with respect to input: ", grad_input)
    #     print("Gradient with respect to output: ", grad_output)

    def forward(self, inp):
        y = inp
        y = self.attention_group_norm(y)
        y = self.proj_in(y)
        y = rearrange(y, 'b c h w -> b (h w) c')
        return y

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LinearAttentionNAFBlock(nn.Module): # bad name, which is not really linear, we would change this name later. if we are fortune enough, it would be our baseline 
    def __init__(self, c, attention_tans_block: LinearAttentionTransBlock, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., context_dim=768,num_heads = 8):
        super().__init__()
        dw_channel = c * DW_Expand
        self.chan = c
        self.attention_tans_block = attention_tans_block
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
        self.norm3 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout3 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros((1, 1, c)), requires_grad=True)
        
        # num_heads = 8
        dim_head = c // num_heads
        inner_dim = num_heads * dim_head
        self.attention_layer_norm1 = nn.LayerNorm(inner_dim)
        # self.proj_out = zero_module(nn.Conv2d(inner_dim,
        #                                       c,
        #                                       kernel_size=1,
        #                                       stride=1,
        #                                       padding=0))
        
        self.attn2 = LightweightLinearCrossAttention(query_dim=inner_dim, context_dim=context_dim,
                                    heads=num_heads, dim_head=dim_head, dropout=drop_out_rate)  # is self-attn if context is none
        

    def forward(self, inp, context):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x* self.beta
        
        b, c, h, w = y.shape
        y = self.attention_tans_block(y)
        x_in = y
        
        y = self.dropout2(self.attn2(self.attention_layer_norm1(y), context=context)) + x_in#self.alpha * x_in
        x = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
        x_in = x
        
        x = self.conv4(self.norm3(x))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout3(x)

        return x_in + x * self.gamma

class LinearAttentionTimeEmbedNAFNet(nn.Module): # bad name, which mean it is not really linear, we would change the name later

    def __init__(self, 
                 img_channel=3, 
                 width=16, 
                 middle_blk_num=1, 
                 org_time_embed_dim = 320, 
                 prompt_embed_max_len = 77,
                 prompt_embed_dim = 768,
                 enc_blk_nums=[], 
                 dec_blk_nums=[],
                 use_time_embed = True):
        super().__init__()
        
        # self.prompt_project = nn.Conv1d(prompt_embed_max_len, prompt_embed_max_len + (prompt_embed_max_len % 2), kernel_size=1) 
        self.use_time_embed = use_time_embed

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.enc_trans_blks = nn.ModuleList()
        self.mid_trans_blks = nn.ModuleList()
        self.dec_trans_blks = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.org_time_embed_dim = org_time_embed_dim
        self.time_embed_dim = self.org_time_embed_dim * 4
        
        self.time_embed = nn.Sequential(
            nn.Conv2d(in_channels=self.org_time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True),
            SimpleGate(),
            nn.Conv2d(in_channels=self.time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True)
        )

        chan = width
        idx = 0
        for num in enc_blk_nums:
            self.enc_trans_blks.extend(
                [LinearAttentionTransBlock(c=chan,num_heads=1) for _ in range(2)]
            )
            self.encoders.append(
                TimestepEmbedSequential(
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    # LinearAttentionTransBlock(c=chan),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.enc_trans_blks[idx],context_dim=prompt_embed_dim,num_heads=1) for _ in range(num // 2)],
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.enc_trans_blks[idx + 1],context_dim=prompt_embed_dim,num_heads=1) for _ in range(num // 2)]
                )
            )
            idx += 2
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        self.mid_trans_blks.extend(
                [LinearAttentionTransBlock(c=chan,num_heads=1) for _ in range(2)]
            )
        idx = 0
        self.middle_blks = \
            TimestepEmbedSequential(
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    # LinearAttentionTransBlock(c=chan),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.mid_trans_blks[idx],context_dim=prompt_embed_dim,num_heads=2) for _ in range(num // 2)],
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.mid_trans_blks[idx + 1],context_dim=prompt_embed_dim,num_heads=2) for _ in range(num // 2)]
                )
        idx = 0
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.dec_trans_blks.extend(
                [LinearAttentionTransBlock(c=chan,num_heads=1) for _ in range(2)]
            )
            self.decoders.append(
                TimestepEmbedSequential(
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    # LinearAttentionTransBlock(c=chan),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.dec_trans_blks[idx],context_dim=prompt_embed_dim,num_heads=1) for _ in range(num // 2)],
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[LinearAttentionNAFBlock(c=chan,attention_tans_block=self.dec_trans_blks[idx + 1],context_dim=prompt_embed_dim,num_heads=1) for _ in range(num // 2)]
                )
            )
            idx += 2

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, timesteps=torch.tensor([1000.0]), prompt_emb:torch.tensor = None,idx=None):
        # print(inp.size())
        # print(inp.shape)
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        timesteps = timesteps.flatten()
        # prompt_emb = self.prompt_project(prompt_emb)

        encs = []
        # print('timesteps', timesteps)
        t_emb = timestep_embedding(timesteps, self.org_time_embed_dim, repeat_only=False)
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]
        # print('t_emb',t_emb.shape)
        idx_emb = guidance_scale_embedding(w=idx,embedding_dim=t_emb.shape[1]).to(device=x.device)
        
        # print('t_emb',t_emb.shape)
        emb = self.time_embed(t_emb.to(device=x.device))
        emb = emb + idx_emb

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x,emb,prompt_emb)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x,emb,prompt_emb)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x,emb,prompt_emb)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
class TimeEmbedLinearAttentionNAFBlock(TimestepBlock):
    def __init__(self, c, prompt_channel_width = 768, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
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
        self.sxa_kv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=prompt_channel_width, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            SimpleGate(),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sxa_q = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
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
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, time_emb, prompt_emb):
        x = inp
        # prompt_emb.unsqueeze(dim=-1)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        
        prompt_kv_value = self.sxa_kv(prompt_emb)
        x = self.sxa_q(y)
        x_out = x * prompt_kv_value
        y = x_out *self.alpha + y

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimeEmbedPureXABNAFBlock(TimestepBlock):
    def __init__(self, c, prompt_channel_width = 768, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sxa_kv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=prompt_channel_width, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            SimpleGate(),
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
        self.norm3 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        self.scale_shift_table = nn.Parameter(torch.randn(6, c) / c ** 0.5)

    def forward(self, inp, t, prompt_emb):

        B, N, C = inp.shape
        needs_reshape = inp.dim() == 3
        if needs_reshape:
            b, seq_len, c = inp.shape
            spatial = int(math.sqrt(seq_len))
            assert spatial * spatial == seq_len, f"Sequence length {seq_len} is not a perfect square"
            inp = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)
      
        inp = self.norm1(inp)
        inp = rearrange(inp, 'b c h w -> b (h w) c')
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        inp = t2i_modulate(inp, shift_msa, scale_msa).reshape(B, N, C)
        inp_4d = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)

        # needs_reshape = inp.dim() == 3

        # if needs_reshape:
        #     b, seq_len, c = inp.shape
        #     spatial = int(math.sqrt(seq_len))
        #     assert spatial * spatial == seq_len, f"Sequence length {seq_len} is not a perfect square"
        #     inp_4d = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)
        # else:
        #     inp_4d = inp

        x = inp_4d
        # prompt_emb.unsqueeze(dim=-1)

        x = self.norm2(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        prompt_emb = prompt_emb.view(B, C, 7, -1)
        prompt_kv_value = self.sxa_kv(prompt_emb)
        x = x * prompt_kv_value
        # x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp_4d + x * self.beta

        x = rearrange(self.norm3(y), 'b c h w -> b (h w) c')
        x = t2i_modulate(x, shift_mlp, scale_mlp).reshape(B, N, C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=spatial, w=spatial)

        x = self.conv4(x) 
        x = self.sg(x)
        x = rearrange(x, 'b c h w -> b (h w) c') * gate_mlp
        x = rearrange(x, 'b (h w) c -> b c h w', h=spatial, w=spatial)
        x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x
    
class SimpliestTimeEmbedPureXABNAFBlock(TimestepBlock):
    def __init__(self, c, prompt_channel_width = 768, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., head_nums=4):
        super().__init__()
        dw_channel = c * DW_Expand
        self.head_nums = head_nums
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # self.sxa_kv = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=prompt_channel_width, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True)
        # )

        assert (dw_channel // 2) % self.head_nums == 0, "Channel count must be divisible by head_nums for SXA"
        per_head_inp_channels = (prompt_channel_width) // self.head_nums
        per_head_out_channels = (dw_channel // 2) // self.head_nums
        self.sxa_kv_mh = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=per_head_inp_channels, out_channels=per_head_out_channels, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            ) for _ in range(self.head_nums)
        ])

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm3 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        self.scale_shift_table = nn.Parameter(torch.randn(6, c) / c ** 0.5)

    def forward(self, inp, t, prompt_emb):

        B, N, C = inp.shape
        needs_reshape = inp.dim() == 3
        if needs_reshape:
            b, seq_len, c = inp.shape
            spatial = int(math.sqrt(seq_len))
            assert spatial * spatial == seq_len, f"Sequence length {seq_len} is not a perfect square"
            inp = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)
      
        inp = self.norm1(inp)
        inp = rearrange(inp, 'b c h w -> b (h w) c')
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        inp = t2i_modulate(inp, shift_msa, scale_msa).reshape(B, N, C)
        inp_4d = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)

        # needs_reshape = inp.dim() == 3

        # if needs_reshape:
        #     b, seq_len, c = inp.shape
        #     spatial = int(math.sqrt(seq_len))
        #     assert spatial * spatial == seq_len, f"Sequence length {seq_len} is not a perfect square"
        #     inp_4d = rearrange(inp, 'b (h w) c -> b c h w', h=spatial, w=spatial)
        # else:
        #     inp_4d = inp

        x = inp_4d
        # prompt_emb.unsqueeze(dim=-1)

        x = self.norm2(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        prompt_emb = prompt_emb.view(B, C, 7, -1)
        
        if self.head_nums > 1:
            head_chunks = torch.chunk(prompt_emb, self.head_nums, dim=1)
            attended = [ kv(chunk) for chunk, kv in zip(head_chunks, self.sxa_kv_mh)]
            prompt_kv_value = torch.cat(attended, dim=1)
        else:
            prompt_kv_value = self.sxa_kv_mh[0](x)

        # prompt_kv_value = self.sxa_kv(prompt_emb)
        x = x * prompt_kv_value
        # x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp_4d + x * self.beta

        x = rearrange(self.norm3(y), 'b c h w -> b (h w) c')
        x = t2i_modulate(x, shift_mlp, scale_mlp).reshape(B, N, C)
        x = rearrange(x, 'b (h w) c -> b c h w', h=spatial, w=spatial)

        x = self.conv4(x) 
        x = self.sg(x)
        x = rearrange(x, 'b c h w -> b (h w) c') * gate_mlp
        x = rearrange(x, 'b (h w) c -> b c h w', h=spatial, w=spatial)
        x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x
    
    
    
class TimeEmbedWithSimplifiedXABNAFNet(nn.Module):

    def __init__(self
                 , img_channel=3
                 , width=16
                 , middle_blk_num=1
                 , org_time_embed_dim = 320
                 , enc_blk_nums=[]
                 , dec_blk_nums=[] 
                 , use_time_embed = True
                 , prompt_channel_width=768):
        super().__init__()
        
        self.use_time_embed = use_time_embed
        self.prompt_channel_width = prompt_channel_width

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.org_time_embed_dim = org_time_embed_dim
        self.time_embed_dim = self.org_time_embed_dim * 4
        
        self.time_embed = nn.Sequential(
            nn.Conv2d(in_channels=self.org_time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True),
            SimpleGate(),
            nn.Conv2d(in_channels=self.time_embed_dim, out_channels=self.time_embed_dim * 2, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True)
        )

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                TimestepEmbedSequential(
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(num // 2)],
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(num // 2)],
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            TimestepEmbedSequential(
                TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(middle_blk_num // 2)],
                TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(middle_blk_num // 2)],
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
                TimestepEmbedSequential(
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(num // 2)],
                    TimeEmbedResBlock(c=chan,org_time_embed_dim=self.org_time_embed_dim),
                    *[TimeEmbedLinearAttentionNAFBlock(chan,prompt_channel_width=self.prompt_channel_width) for _ in range(num // 2)],
                )
            )
        
        self.last_layer = list(self.decoders[-1].children())[-1]
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, timesteps=torch.tensor([1000.0]), prompt_emb: torch.tensor = None,idx: torch.tensor = None):
        # print(inp.size())
        # print(inp.shape)
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        prompt_emb = rearrange(prompt_emb.unsqueeze(dim=1), 'b h w c -> b c w h')
        # print(prompt_emb.shape)

        x = self.intro(inp)
        timesteps = timesteps.flatten()

        encs = []
        # print('timesteps', timesteps)
        t_emb = timestep_embedding(timesteps, self.org_time_embed_dim, repeat_only=False)
        if idx is not None:
            idx_emb = guidance_scale_embedding(w=idx,embedding_dim=t_emb.shape[1]).to(device=x.device)
        else:
            idx_emb = guidance_scale_embedding(w=timesteps,embedding_dim=t_emb.shape[1]).to(device=x.device)
        
        # print('t_emb',t_emb.shape)
        t_emb = t_emb + idx_emb
        while len(t_emb.shape) < len(x.shape):
            t_emb = t_emb[..., None]
        emb = self.time_embed(t_emb.to(device=x.device))
        # emb = emb + idx_emb

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x,emb,prompt_emb)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x,emb,prompt_emb)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x,emb,prompt_emb)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

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