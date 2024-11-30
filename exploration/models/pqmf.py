"""https://github.com/acids-ircam/RAVE/blob/master/rave/pqmf.py"""
"""Pseudo QMF modules."""
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.optimize import fmin
from scipy.signal import firwin, kaiserord


def reverse_half(x):
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1

    return x * mask


def center_pad_next_pow_2(x):
    next_2 = 2**math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))


def make_odd(x):
    if not x.shape[-1] % 2:
        x = nn.functional.pad(x, (0, 1))
    return x


def get_qmf_bank(h, n_band):
    """
    Modulates an input protoype filter into a bank of
    cosine modulated filters
    Parameters
    ----------
    h: torch.Tensor
        prototype filter
    n_band: int
        number of sub-bands
    """
    k = torch.arange(n_band).reshape(-1, 1)
    N = h.shape[-1]
    t = torch.arange(-(N // 2), N // 2 + 1)

    p = (-1)**k * math.pi / 4

    mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)
    hk = 2 * h * mod

    return hk


def kaiser_filter(wc, atten, N=None):
    """
    Computes a kaiser lowpass filter
    Parameters
    ----------
    wc: float
        Angular frequency
    
    atten: float
        Attenuation (dB, positive)
    """
    N_, beta = kaiserord(atten, wc / np.pi)
    N_ = 2 * (N_ // 2) + 1
    N = N if N is not None else N_
    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)
    return h


def loss_wc(wc, atten, M, N):
    """
    Computes the objective described in https://ieeexplore.ieee.org/document/681427
    """
    h = kaiser_filter(wc, atten, N)
    g = np.convolve(h, h[::-1], "full")
    g = abs(g[g.shape[-1] // 2::2 * M][1:])
    return np.max(g)


def get_prototype(atten, M, N=None):
    """
    Given an attenuation objective and the number of bands
    returns the corresponding lowpass filter
    """
    wc = fmin(lambda w: loss_wc(w, atten, M, N), 1 / M, disp=0)[0]
    return kaiser_filter(wc, atten, N)


def polyphase_forward(x, hk, rearrange_filter=True):
    """
    Polyphase implementation of the analysis process (fast)
    Parameters
    ----------
    x: torch.Tensor
        signal to analyse ( B x 1 x T )
    
    hk: torch.Tensor
        filter bank ( M x T )
    """
    x = rearrange(x, "b c (t m) -> b (c m) t", m=hk.shape[0])
    if rearrange_filter:
        hk = rearrange(hk, "c (t m) -> c m t", m=hk.shape[0])
    x = nn.functional.conv1d(x, hk, padding=hk.shape[-1] // 2)[..., :-1]
    return x


def polyphase_inverse(x, hk, rearrange_filter=True):
    """
    Polyphase implementation of the synthesis process (fast)
    Parameters
    ----------
    x: torch.Tensor
        signal to synthesize from ( B x 1 x T )
    
    hk: torch.Tensor
        filter bank ( M x T )
    """

    m = hk.shape[0]

    if rearrange_filter:
        hk = hk.flip(-1)
        hk = rearrange(hk, "c (t m) -> m c t", m=m)  # polyphase

    pad = hk.shape[-1] // 2 + 1
    x = nn.functional.conv1d(x, hk, padding=int(pad))[..., :-1] * m

    x = x.flip(1)
    x = rearrange(x, "b (c m) t -> b c (t m)", m=m)
    x = x[..., 2 * hk.shape[1]:]
    return x

@torch.fx.wrap
class PQMF(nn.Module):
    """
    Pseudo Quadrature Mirror Filter multiband decomposition / reconstruction
    Parameters
    ----------
    attenuation: int
        Attenuation of the rejected bands (dB, 80 - 120)
    n_band: int
        Number of bands, must be a power of 2 if the polyphase implementation
        is needed
    """

    def __init__(self, attenuation, n_band, polyphase=True, n_channels = 1):
        super().__init__()
        h = get_prototype(attenuation, n_band)

        if polyphase:
            power = math.log2(n_band)
            assert power == math.floor(
                power
            ), "when using the polyphase algorithm, n_band must be a power of 2"

        h = torch.from_numpy(h).float()
        hk = get_qmf_bank(h, n_band)
        hk = center_pad_next_pow_2(hk)

        self.register_buffer("hk", hk)
        self.register_buffer("h", h)
        self.n_band = n_band
        self.polyphase = polyphase
        self.n_channels = n_channels

    def forward(self, x):
        if x.ndim == 2:
            return torch.stack([self.forward(x[i]) for i in range(x.shape[0])])
        if self.n_band == 1:
            return x
        x = polyphase_forward(x, self.hk.to(x.device))

        x = reverse_half(x)

        return x

    def inverse(self, x):
        if x.ndim == 2:
            if self.n_channels == 1:
                return self.inverse(x[0]).unsqueeze(0)
            else:
                x = x.split(self.n_channels, -2)
                return torch.stack([self.inverse(x[i]) for i in len(x)])

        if self.n_band == 1:
            return x

        x = reverse_half(x)

        return polyphase_inverse(x, self.hk)
