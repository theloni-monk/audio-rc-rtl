import math
import random

import torch
from torch import nn

# mel scale
def mels2freqrange(n_freqs, min_mel, max_mel):
    return 700*((10**(torch.linspace(min_mel, max_mel, n_freqs)/2950))-1)

def make_sinc(cutoff, size):
    half = size//2
    idx = torch.linspace(-half, half, size)

    filt = torch.special.sinc(cutoff * idx)
    filt = filt * torch.hamming_window(size, periodic=False)
    filt = filt / filt.sum().abs()
    return filt

def make_bpasses(n_channels, kwidth = 133, min_mel = 10, max_mel = 4000, srate=44100):
    w = torch.zeros(n_channels, kwidth)
    half = kwidth // 2
    freqs = mels2freqrange(n_channels+1, min_mel, max_mel)
    for i in range(n_channels):
    
        Fcl = freqs[i] / srate
        Wl = 2 * math.pi * Fcl
        
        Fch = freqs[i+1] / srate
        Wh = 2 * math.pi * Fch
        
        filt = make_sinc(Wl, kwidth) - make_sinc(Wh, kwidth)
        w[i] = filt / torch.fft.rfft(filt).abs()[1:].max()

    return w

class FIRBlock(nn.Module):
    def __init__(self, kernel_width, degree, channels_in, channels_out):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=1.0/16)
        self.c1d = nn.Conv1d(channels_in, channels_out, kernel_width, groups=channels_out // degree, padding='same', bias=False)
        self.norm = nn.InstanceNorm1d(channels_out, affine=True)
        with torch.no_grad():
            bps = make_bpasses(channels_out)
            self.c1d.weight.data = 

    @staticmethod
    def to_mel(hz):
        return 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def forward(self, x):
        return self.lrelu(self.norm(self.c1d(x)))
    

