import sys
sys.path.append('./src')
import torch
import torch.nn as nn
from seqmodel.seq.transform import complement, reverse, reverse_complement


class RCIConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                do_reverse=True, do_complement=True):

        transforms = [super().forward]
        divisor = 1
        if do_reverse:
            transforms += [self.reverse]
        if do_complement:
            divisor = 2
            transforms += [self.complement]
        if do_reverse and do_complement:
            divisor = 4
            transforms += [self.reverse_complement]
        assert out_channels % divisor == 0

        super().__init__(in_channels, out_channels // divisor, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._transforms = transforms
        self.out_channels = out_channels

    def complement(self, x):
        return self(complement(x))

    def reverse(self, x):
        return reverse(self(reverse(x)))

    def reverse_complement(self, x):
        return reverse(self(reverse_complement(x)))

    def forward(self, x):
        return torch.cat([t(x) for t in self._transforms], dim=1)


# stack outputs of arbitrary modules along dimension
class CombineChannels(nn.Module):

    def __init__(self, *layer_list, stack_dim=1):
        self.layer_list = layer_list
        self.stack_dim = stack_dim

    def forward(self, x):
        return combine_channels([layer(x) for layer in self.layer_list], stack_dim=self.stack_dim)


# uses pooling
class RCIConvUnit(nn.Conv1d):

    def __init__(self, in_channels, out_channels,
                stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return combine_outputs(
                super().forward(x),
                super().forward(torch.flip(x, [1]))
            )
