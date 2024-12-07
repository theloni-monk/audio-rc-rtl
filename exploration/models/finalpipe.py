import torch
from torch import nn
import torch.nn.functional as F
from models.lfsr import LFSR
from models.pqmf import PQMF

def mtmt(x, lin):
    return lin(x.mT).mT

def nonlin(x):
    return torch.clamp(4*x, -1.0, 1.0)

def reluleak128(x):
    return F.leaky_relu(x, 1.0/128)


class LowBit_RC(nn.Module):
    def __init__(self):
        self.n_bits = 4
        self.seed = 69

        super().__init__()

    def _gen_weights(self):
        # cyclemask = sum(torch.eye(self.in_dim).roll(i, dims=0) for i in range(1, self.degree)).bool()
        lfsr = LFSR(seed = self.seed)
        A = torch.tensor([lfsr.gen_fxp_shift(self.n_bits) for _ in range(64)]).reshape(8,8)
        return 0.1* A / torch.linalg.eigvals(A)[0].abs()

class FSDDRC(LowBit_RC):
    def __init__(self):
        super().__init__()
        self.feature_pushforward = self._gen_weights()

    def forward(self, x): # x: [batch, vector_dim, time]
        features = [torch.zeros(x.shape[0], 8, 1, device=x.device)]
        out = []
        for i in range(1024):
            f = nonlin( self.feature_pushforward.to(x.device) @ features[-1] + x[..., i].unsqueeze(-1))
            out.append(torch.cat((f, reluleak128(f)), dim = -2))
            features.append(f)
        return torch.cat(out,dim = -1)

class TorchPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        bands = 8
        self.whiten = nn.BatchNorm1d(8)
        self.rc = FSDDRC()
        
        self.ff1 = nn.Linear(bands, 3*bands)
        self.ff2 = nn.Linear(3*bands, 2*bands)

        allfeats = 6*bands
        self.lin_out = nn.Linear(allfeats, 11)
        self.preact_norm = nn.BatchNorm1d(11)

    def forward(self, x):

        bwhitened =  self.whiten(x) #(x - self.channel_bias.unsqueeze(-1)) * self.channel_scale.unsqueeze(-1)

        nl_tdyn = self.rc(bwhitened)

        ff = reluleak128(mtmt(reluleak128(mtmt(bwhitened, self.ff1)), self.ff2))
        
        stacked = torch.cat((x, bwhitened, 
                             ff, nl_tdyn), dim=-2)

        assimilated = reluleak128(self.preact_norm(mtmt(stacked, self.lin_out)))

        return  assimilated.mean(dim=-1)