import sys
sys.path.append('./src')
import torch
import torch.nn as nn
from seqmodel.seq.transform import complement, reverse, reverse_complement


class RCIConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                do_reverse=True, do_complement=True):

        transforms = [nn.Identity()]
        if do_reverse and do_complement:
            transforms += [reverse_complement]
        if do_reverse:
            transforms += [reverse]
        if do_complement:
            transforms += [complement]
        assert out_channels % len(transforms) == 0

        super().__init__(in_channels, out_channels // len(transforms), kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._transforms = transforms
        self.out_channels = out_channels

    # output channels are ordered as (normal, reverse_complement, reverse, complement)
    # complementation is still meaningful by flipping (reversing) the channel dimension
    def forward(self, x):
        return torch.cat([t(super(RCIConv1d, self).forward(t(x)))
                        for t in self._transforms], dim=1)
    

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
