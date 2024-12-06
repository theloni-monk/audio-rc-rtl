import math
import torch
from torch import nn
import torch.nn.functional as F

from .lfsr import LFSR
from .pqmf import PQMF
# from ..data.torchfsdd import WAV_LEN
def mtmt(x, lin):
    return lin(x.mT).mT


class Clipper(nn.Module):
    def __init__(self, slope = 2, soft=True):
        super().__init__()
        self.slope = slope
        self.soft = soft

    def forward(self, x):
        x = self.slope*x
        if self.soft:
            return F.tanh(x)
        return torch.clamp(x, -1.0, 1.0)

# if your nonlinearity isn't in the feedback path you should probably just use a linear reservoir tbh
class Abstract_RC(nn.Module):
    def __init__(self, in_dim, out_dim, nonlin = Clipper(), normalizer = None, skip_lin = False, symbreak=None):
        self.skip_lin = skip_lin
        super().__init__()
        self.in_dim = in_dim
        if symbreak is not None:
            in_dim *= 2
        self.nonlin = nonlin
        if not skip_lin:
            self.lin_out = nn.Linear(in_dim, out_dim)
        self.feature_pushforward = self._gen_weights()
        self.norm = normalizer
        self.symbreak = symbreak

    def update_feats(self, datavec):
        self.features = self.features.to(datavec.device)
        self.features @= self.feature_pushforward.to(datavec.device)
        self.features = self.features.expand(datavec.shape) + datavec
        self.features = self.nonlin(self.features)

    def forward(self, x, init_feat = None): # x: [batch, vector_dim, time]

        self.features = torch.zeros(x.shape[:-1]) if init_feat is None else init_feat
        out = []
        for i in range(x.shape[-1]):
            self.update_feats(x[..., i])
            out.append(self.features)

        resbuf =  torch.stack(out).movedim(0, -1)
        if self.norm is not None:
            resbuf = self.norm(resbuf)
        if self.symbreak is not None:
            resbuf = torch.cat((resbuf, self.symbreak(resbuf)), dim=1)
        if self.skip_lin:
            return resbuf
        return mtmt(resbuf, self.lin_out)

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

        cyclemask = sum(torch.eye(self.f_out, self.f_in).roll(i, dims=0) 
                        for i in range(1, self.degree)).bool() if self.degree > 1 else None
        lfsr = LFSR(seed = self.seed)
        self.mixer = torch.zeros(self.f_out, self.f_in)
        if cyclemask is not None:
            self.mixer[cyclemask] = torch.tensor([lfsr.gen_fxp_shift(self.n_bits) for _ in range(cyclemask.sum())])
        else:
            self.mixer = torch.tensor([lfsr.gen_fxp_shift(self.n_bits) for _ in range(self.mixer.numel())]).unsqueeze(-1)
        
        # torch.random.manual_seed(self.seed)
        # self.permutation = torch.randperm(self.f_in)
        # torch.random.seed()

    def forward(self, x):
        return self.mixer.to(x.device) @ x #[:, self.permutation, :]

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

def reluleak64(x):
    return F.leaky_relu(x, 1.0/64)

def reluleak128(x):
    return F.leaky_relu(x, 1.0/128)

N_CLASSES = 11
class FSDDPipelineV8(nn.Module):
    def __init__(self, n_bands, n_bits, spec_rad, rc_multiplex_degree, pl_version = 8.0, seed = 69):
        super().__init__()

        self.nb = n_bands
        rc_nl =  Clipper(slope=4, soft=False)
        self.rc = LowBit_RC(n_bits, seed, 
                            rc_multiplex_degree, spec_rad, 
                            n_bands, n_bands, 
                            rc_nl,  None, #nn.InstanceNorm1d(n_bands), 
                            True, reluleak128) # symbreaking
        # self.linrc = LowBit_RC(n_bits, 2*seed-1, 
        #                     rc_multiplex_degree, spec_rad, 
        #                     n_bands, n_bands, 
        #                     nn.Identity(),  None, #nn.InstanceNorm1d(n_bands), 
        #                     True, reluleak128) # symbreaking

        
        self.band_norm = nn.InstanceNorm1d(n_bands)
        
        self.ff1 = nn.Linear(n_bands, 3*n_bands)
        self.ff2 = nn.Linear(3*n_bands, 2*n_bands)

        allfeats = 6*n_bands
        self.lin_out = nn.Linear(allfeats, N_CLASSES)

        self.preact_norm = nn.BatchNorm1d(N_CLASSES)

    def forward(self, bands):
        nbands = self.band_norm(bands)

        nl_tdyn = self.rc(nbands)
        # l_tdyn = self.linrc(nbands)

        ff = reluleak128(mtmt(reluleak128(mtmt(nbands, self.ff1)), self.ff2))
        
        stacked = torch.cat((bands, nbands, 
                             ff, nl_tdyn), dim=1)
        assimilated = reluleak128(self.preact_norm(mtmt(stacked, self.lin_out)))

        return  assimilated.mean(dim=-1)
