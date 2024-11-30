import operator
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

from scipy.signal import firwin

from .rc import *

def prod(nums):
    return reduce(operator.mul, nums) if len(nums) > 1 else nums[0] if len(nums) == 1 else 1

def aa_fir(dec_factor):
    half_len = 10 * dec_factor  # reasonable cutoff for our sinc-like function
    n = 2 * half_len
    b, a = firwin(n+1, 1. / dec_factor, window='hamming'), 1.
    return b

def crosscorr(x, k):
    return F.conv1d(x.unsqueeze(0), k.to(x.device), padding='same')

#TODO: make filter half(1/q) band to be more efficient
class DownSampler(nn.Module):
    def __init__(self, dec_factor, channels):
        super().__init__()
        self.dec_fac = dec_factor
        self.antialias = torch.flip(torch.tensor(aa_fir(dec_factor)), (0,)).repeat(channels, channels, 1).float()
        # self.apply = torch.vmap(torch.vmap(crosscorr, in_dims=(0, None)), in_dims=(0, None))

    def forward(self, x):
        return F.conv1d(x, self.antialias.to(x.device), padding='same')[..., torch.arange(0, x.shape[-1], self.dec_fac)] #self.apply(x, self.antialias).flatten(-2, -1)[..., torch.arange(0, x.shape[-1], self.dec_fac)]


# TODO: better defined link functions
# TODO: PC inference between layers
class HierarchicalNonlinTSModel(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, base_samplerate,  feature_reductions, rate_reductions, proto_model, model_params):
        super().__init__()
        
        assert len(feature_reductions) ==  len(rate_reductions), "inequal number of feature and rate reductions"

        self.out_rate = base_samplerate / prod(rate_reductions)

        self.models = nn.ModuleList([proto_model(   *model_params, 
                                                    int(round(in_feat_dim / prod(feature_reductions[:i]))), 
                                                    int(round(in_feat_dim / prod(feature_reductions[:i+1]))) if i < len(feature_reductions) - 1 else out_feat_dim, 
                                                  ) 
                                         for i in range(len(feature_reductions))])

        self.downsamplers = nn.ModuleList([DownSampler(rr, 
                                                       int(round(in_feat_dim / prod(feature_reductions[:i+1])))
                                                       if i < len(feature_reductions) - 1 else out_feat_dim)
                                           for i, rr in enumerate(rate_reductions)])

    def forward(self, x):
        # buffers = [x]
        for rc, ds in zip(self.models, self.downsamplers):
            # buffers.append(ds(rc(buffers[-1])))
            x = ds(rc(x))
        return F.relu(x) #HACK want relu for classification with crosent
