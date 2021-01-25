import sys
sys.path.append('./src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from seqmodel.functional import complement, reverse, reverse_complement, swap


class GroupConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, in_from_group=False,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                do_reverse=True, do_complement=True):

        super(GroupConv1d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kernel_size = (kernel_size, 1)
        self.w = nn.Parameter(self.weight.view(self.weight.size(0), -1).t())

    # output channels are ordered as (normal, reverse_complement, reverse, complement)
    # complementation is still meaningful by flipping (reversing) the channel dimension
    def forward(self, x):
        unfolded_x = F.unfold(x.view(*x.shape, 1), (self.kernel_size[0], 1)).transpose(1, 2)
        # w = self.weight.view(self.weight.size(0), -1).t()
        y = unfolded_x.matmul(self.w)
        
        if not (self.bias is None):
            y = y + self.bias
        return y.transpose(1, 2)


class v2GroupConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, in_from_group=False,
                stride=1, padding=0, dilation=1, groups=1, bias=True):

        super(v2GroupConv1d, self).__init__(in_channels, out_channels // 4, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.in_from_group = in_from_group

    # output channels are ordered as (normal, reverse_complement, reverse, complement)
    # complementation is still meaningful by flipping (reversing) the channel dimension
    def forward(self, x):
        if self.in_from_group:
            x_all = torch.cat(
                [
                    x,
                    swap(reverse_complement(x)),
                    reverse(swap(x)),
                    complement(x),
                ], dim=0)
        else:
            x_all = torch.cat(
                [
                    x,
                    reverse_complement(x),
                    reverse(x),
                    complement(x),
                ], dim=0)
        y_all = super(v2GroupConv1d, self).forward(x_all)
        y = y_all.view(x.shape[0], self.out_channels * 4, -1)
        y, y_rc, y_r, y_c = torch.split(y, y.shape[1] // 4, dim=1)
        return torch.cat([
                y,
                reverse_complement(y_rc),
                reverse(y_r),
                complement(y_c),
            ])


class RCIConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, in_from_rci=False,
                stride=1, padding=0, dilation=1, groups=1, bias=True,
                do_reverse=True, do_complement=True):

        t_in, t_out = [nn.Identity()], [nn.Identity()]
        if do_reverse and do_complement:
            if in_from_rci:
                t_in += [lambda x: reverse_complement(swap(x))]
            else:
                t_in += [reverse_complement]
            t_out += [reverse_complement]
        if do_reverse:
            if in_from_rci:
                t_in += [lambda x: reverse(swap(x))]
            else:
                t_in += [reverse]
            t_out += [reverse]
        if do_complement:
            t_in += [complement]
            t_out += [complement]
        assert out_channels % len(t_in) == 0

        super().__init__(in_channels, out_channels // len(t_in), kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self._transforms_in = t_in
        self._transforms_out = t_out

    # output channels are ordered as (normal, reverse_complement, reverse, complement)
    # complementation is still meaningful by flipping (reversing) the channel dimension
    def forward(self, x):
        return torch.cat([t_out(super(RCIConv1d, self).forward(t_in(x)))
                        for t_in, t_out in zip(self._transforms_in, self._transforms_out)], dim=1)
    

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
