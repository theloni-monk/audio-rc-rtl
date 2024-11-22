import torch
from torch import nn
import torch.nn.functional as F

from .lfsr import Int8_LFSR

# based on nlin,https://www.nature.com/articles/s41598-023-39886-w
class Abstract_RC(nn.Module):
    def __init__(self, feat_dim, out_shape, nonlin = lambda x: min(1, max(-1, x))):
        self.n_feat = feat_dim
        self.out_buffer = torch.zeros(out_shape)
        self.nonlin = nonlin
        self.w_out = nn.Parameter(0.1*torch.randn(self.out_shape[1], feat_dim))
        self.feature_pushforward = self._gen_weights()

    def update_feats(self, datavec):
        self.features = self.nonlin((self.feature_pushforward @ self.features) + datavec)

    def forward(self, x): # x: [batch, vector_dim, time]
        assert x.shape[-1] == self.out_buffer.shape[-1], "preallocated outbuffer not aligned to input"
        self.out_buffer = self.out_buffer.to(x.device)
        for i in range(self.in_shape[-1]):
            self.update_feats(x[..., i])
            self.out_buffer[..., i] = (self.w_out.to(x.device) @ self.features.to(x.device))
        return self.out_buffer

class CyclicMultiplex_RC(Abstract_RC):
    def __init__(self, feat_dim, out_shape, degree, spec_rad=0.1, nonlin = lambda x:x):
        super().__init__(feat_dim, out_shape)
        self.degree = degree
        self.spec_rad = spec_rad
        self.nl = nonlin
        self.feature_pushforward = self._gen_weights()

    def _gen_weights(self):
        cyclemask = sum(torch.eye(self.n_feat).roll(i, dims=0) for i in range(1, self.degree))
        A = torch.randn(self.n_feat, self.n_feat)*cyclemask
        return self.spec_rad * A / torch.linalg.eigvals(A)[0]

class SPCTRE_RC(CyclicMultiplex_RC):
    def __init__(self, feat_dim, out_shape, degree, spec_rad=0.1, nonlin=lambda x: x):
        super().__init__(feat_dim, out_shape, degree, spec_rad, nonlin)

    def _gen_weights(self):
        cyclemask = super()._gen_weights() != 0
        lfsr = Int8_LFSR()
        A = cyclemask*lfsr.gen_n_int8s(cyclemask.numel()).float().reshape(cyclemask.shape)
        return self.spec_rad * A / torch.linalg.eigvals(A)[0]