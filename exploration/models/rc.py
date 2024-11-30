import math
import torch
from torch import nn
import torch.nn.functional as F

from .lfsr import LFSR
from .pqmf import PQMF

def clip_nl(x):
    return torch.clamp(4*x, -1, 1) #HACK

# if your nonlinearity isn't in the feedback path you should probably just use a linear reservoir tbh
class Abstract_RC(nn.Module):
    def __init__(self, in_dim, out_dim, nonlin = clip_nl, feedback_nl = True):
        super().__init__()
        self.in_dim = in_dim
        self.feedback_nl = feedback_nl
        self.nonlin = nonlin
        self.lin_out = nn.Linear(in_dim if feedback_nl else 2*in_dim, out_dim)
        self.feature_pushforward = self._gen_weights()

    def update_feats(self, datavec):
        self.features = self.features.to(datavec.device)
        self.features @= self.feature_pushforward.to(datavec.device)
        self.features += datavec
        if self.feedback_nl:
            self.features = self.nonlin(self.features)

    def forward(self, x, init_feat = None): # x: [batch, vector_dim, time]

        self.features = torch.zeros(x.shape[:-1]) if init_feat is None else init_feat
        out = []
        for i in range(x.shape[-1]):
            self.update_feats(x[..., i])
            out.append(self.features.to(x.device))

        resbuf =  torch.stack(out).movedim(0, -1)
        if not self.feedback_nl: # nonlinearity is feed forward
            resbuf = torch.cat((resbuf, self.nonlin(resbuf)), dim=1)

        return self.lin_out(resbuf.mT).mT

class CyclicMultiplex_RC(Abstract_RC):
    def __init__(self, degree, spec_rad=0.1, *args):
        self.degree = degree
        self.spec_rad = spec_rad
        super().__init__(*args)

    def _gen_weights(self):
        cyclemask = sum(torch.eye(self.in_dim).roll(i, dims=0) for i in range(1, self.degree))
        A = torch.randn(self.in_dim, self.in_dim)*cyclemask
        return self.spec_rad * A / torch.linalg.eigvals(A)[0].abs()


class LowBit_RC(CyclicMultiplex_RC):
    def __init__(self, n_bits, seed = 69, *args):
        self.n_bits = n_bits
        self.seed = seed

        super().__init__(*args)

    def _gen_weights(self):
        cyclemask = sum(torch.eye(self.in_dim).roll(i, dims=0) for i in range(1, self.degree)).bool()
        
        lfsr = LFSR(seed = self.seed)
        A = torch.zeros(self.in_dim, self.in_dim)
        A[cyclemask] = torch.tensor([lfsr.gen_fxp_shift(self.n_bits) for _ in range(cyclemask.sum())])

        if self.spec_rad > 0:
            return self.spec_rad * A / torch.linalg.eigvals(A)[0].abs()

        return A

class LowBitMixIn(nn.Module):
    def __init__(self, n_bits, degree, feat_in, feat_out, seed = 69):
        self.n_bits = n_bits
        self.degree = degree
        self.f_in = feat_in
        self.f_out = feat_out

        self.seed = seed
        super().__init__()

        cyclemask = sum(torch.eye(self.f_out, self.f_in).roll(i, dims=0) for i in range(1, self.degree)).bool()
        lfsr = LFSR(seed = self.seed)
        self.mixer = torch.zeros(self.f_out, self.f_in)
        self.mixer[cyclemask] = torch.tensor([lfsr.gen_fxp_shift(self.n_bits) for _ in range(cyclemask.sum())])
        torch.random.manual_seed(self.seed)
        self.permutation = torch.randperm(self.f_in)
    
    def forward(self, x):
        return self.mixer.to(x.device) @ x[:, self.permutation, :]

class FSDDPipeline(nn.Module):
    def __init__(self, n_bands, n_feats, n_bits, spec_rad, mix_degree, rc_multiplex_degree, seed = 69):
        super().__init__()
        
        self.pqmf = PQMF(100, n_bands)

        self.mixin = LowBitMixIn(n_bits, mix_degree, n_bands, n_feats) if mix_degree > 0 else nn.Identity()

        self.rc = LowBit_RC(n_bits, seed, rc_multiplex_degree, spec_rad, n_feats, 10)
        self.ff1 = nn.Linear(n_feats, n_feats//8)
        self.ff2 = nn.Linear(n_feats//8,10)

    def forward(self, x):
        bands = self.pqmf(x)
        mixed = self.mixin(bands)
        res = F.leaky_relu(self.rc(mixed), 1.0/16)
        ff = F.leaky_relu(self.ff2(F.leaky_relu(self.ff1(mixed.mT), 1.0/16)), 1.0/16).mT
        
        return  F.relu(res + ff).mean(dim=-1)

class FSDDPipelineV2(nn.Module):
    def __init__(self, n_bands, n_feats, n_bits, spec_rad, mix_degree, rc_multiplex_degree, seed = 69):
        super().__init__()

        self.mixin = LowBitMixIn(n_bits, mix_degree, n_bands, n_feats) if mix_degree > 0 else nn.Identity()

        self.rc = LowBit_RC(n_bits, seed, rc_multiplex_degree, spec_rad, n_feats, 10)
        self.rc_bypass = nn.Linear(n_feats, 10)
        
        self.ff1 = nn.Linear(n_bands, n_bands//2)
        self.ff2 = nn.Linear(n_bands//2,10)

    def forward(self, bands):
        mixed = self.mixin(bands)
        res = F.leaky_relu(self.rc(mixed), 1.0/16) + F.leaky_relu(self.rc_bypass(mixed.mT), 1.0/16).mT
        ff = F.leaky_relu(self.ff2(F.leaky_relu(self.ff1(bands.mT), 1.0/16)), 1.0/16).mT
        
        return  F.relu(res + ff).mean(dim=-1)

def reluleak16(x):
    return F.leaky_relu(x, 1.0/16)

def reluleak16r(x):
    return F.leaky_relu(x, -1.0/16)


class FSDDPipelineV3(nn.Module):
    def __init__(self, n_bands, n_feats, n_bits, spec_rad, rc_multiplex_degree, feedback_nl, seed = 69):
        super().__init__()
        self.scramble = LowBitMixIn(n_bits, rc_multiplex_degree, n_bands, n_feats)
        self.rc = LowBit_RC(n_bits, seed, rc_multiplex_degree, spec_rad, n_feats, 10, clip_nl, feedback_nl)
        self.lin_bypass = nn.Linear(n_feats, 10)
        self.ff1 = nn.Linear(n_feats, n_feats//2)
        self.ff2 = nn.Linear(n_feats//2, 10)

    def forward(self, bands):
        scrambled = self.scramble(bands)
        bypass = self.lin_bypass(scrambled.mT).mT
        res = self.rc(scrambled)
        ff = self.ff2(reluleak16(self.ff1(scrambled.mT))).mT
        
        return  reluleak16r(res + bypass + ff).mean(dim=-1)