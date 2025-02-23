from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc
    # https://adefossez.github.io/julius/julius/core.html
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)

def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        act = Activation1d(act)
    
    return act

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
# License available in LICENSES/LICENSE_NVIDIA.txt
class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x

def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)

class Activation1d(nn.Module):

    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None \
            else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size -
                                             self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride +
                                              1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B,C,T]
    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]
        return x


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None \
            else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx
    
#return filter [1,1,kernel_size]
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    #ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter

class LowPassFilter1d(nn.Module):

    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    #input [B,C,T]
    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)
        return out