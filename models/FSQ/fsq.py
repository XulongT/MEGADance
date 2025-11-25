import numpy as np
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encdec import Encoder, Decoder
# from .bottleneck import Bottleneck

def dont_update(params):
    for param in params:
        param.requires_grad = False

def update(params):
    for param in params:
        param.requires_grad = True

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]

def _loss_fn(x_target, x_pred):
    return t.mean(t.abs(x_pred - x_target)) 

class Conv1DEncoder(nn.Module):
    def __init__(self, hps, input_dim=72):
        super().__init__()
        self.levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        self.multipliers = hps.hvqvae_multipliers
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                        dilation_growth_rate=hps.dilation_growth_rate, \
                        dilation_cycle=hps.dilation_cycle, \
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs
        encoder = lambda level: Encoder(input_dim, emb_width, level + 1, downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        for level in range(self.levels):
            self.encoders.append(encoder(level))

    def forward(self, x):
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        return xs

    def preprocess(self, x):
        assert len(x.shape) == 3
        x = x.permute(0,2,1).float()
        return x
        
from einops import rearrange
from vector_quantize_pytorch import ResidualFSQ
class Conv1DQuantizer(nn.Module):
    def __init__(self, hps):
        super().__init__()
        levels = hps.levels
        emb_width = hps.emb_width
        l_bins = hps.l_bins
        mu = hps.l_mu
        # self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        self.bottleneck = ResidualFSQ(
            dim = 512,
            # levels = [8, 5, 5, 5], # 2^10
            levels = [7, 5, 5, 5, 5], # 2^12
            # levels = [8, 8, 8, 6, 5], # 2^14
            # levels = [8, 8, 8, 5, 5, 5], 
            num_quantizers = 1
        )

    def forward(self, xs):
        xs = rearrange(xs[0], 'b c t -> b t c')
        xs_quantised, zs = self.bottleneck(xs)
        xs_quantised = rearrange(xs_quantised, 'b t c -> b c t')
        return [zs], [xs_quantised]

    def get_feature_from_index(self, zs):
        xs_quantised = self.bottleneck.get_output_from_indices(zs)
        xs_quantised = rearrange(xs_quantised, 'b t c -> b c t')
        return [xs_quantised]


class Conv1DDecoder(nn.Module):
    def __init__(self, hps, output_dim=72):
        super().__init__()
        self.levels = hps.levels
        downs_t = hps.downs_t
        strides_t = hps.strides_t
        emb_width = hps.emb_width
        self.multipliers = hps.hvqvae_multipliers 
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv, dilation_growth_rate=hps.dilation_growth_rate, \
                            dilation_cycle=hps.dilation_cycle, reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        decoder = lambda level: Decoder(output_dim, emb_width, level + 1, downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.decoders = nn.ModuleList()
        for level in range(self.levels):
            self.decoders.append(decoder(level))
    
    def forward(self, xs_quantised):
        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)
            x_outs.append(x_out)
        x_out = self.postprocess(x_outs[0])
        return x_out

    def postprocess(self, x):
        x = x.permute(0,2,1)
        return x


class Conv1DNN(nn.Module):
    def __init__(self, hps, input_dim=512, output_dim=512):
        super().__init__()
        self.levels = hps.levels
        self.multipliers = hps.hvqvae_multipliers
        if not hasattr(hps, 'dilation_cycle'):
            hps.dilation_cycle = None
        block_kwargs = dict(down_t=hps.downs_t[0], stride_t=hps.strides_t[0], width=hps.width, depth=hps.depth, m_conv=hps.m_conv, \
                            dilation_growth_rate=hps.dilation_growth_rate, dilation_cycle=hps.dilation_cycle)
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            return this_block_kwargs
        self.NN = NNConvBlock(input_emb_width=input_dim, output_emb_width=output_dim, **_block_kwargs(self.levels))

    def forward(self, x):
        x = x[0]
        return [self.NN(x)]

