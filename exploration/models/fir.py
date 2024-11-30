from torch import nn

class FIRBlock(nn.Module):
    def __init__(self, kernel_width, degree, channels_in, channels_out):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=1.0/16)
        self.c1d = nn.Conv1d(channels_in, channels_out, kernel_width, groups=channels_out // degree, padding='same')
        self.norm = nn.InstanceNorm1d(channels_out, affine=True)

    def forward(self, x):
        return self.lrelu(self.norm(self.c1d(x)))