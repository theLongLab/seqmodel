import sys
sys.path.append('./src')
import torch.nn as nn


class CellModule(nn.Module):

    def __init__(self, layer_list, activation_fn=nn.ReLU):
        super().__init__()
        self.activation_fn = activation_fn
        self.layers = nn.Sequential(*layer_list)

    # number of channels per sequence position
    @property
    def out_channels(self):
        self.layers

    def forward(self, x):
        return self.layers(x)

    def out_seq_len(self, seq_len):
        for module in self.modules():
            if type(module) == nn.Conv1d:
                #FIXME: this calculation is wrong
                seq_len += 2 * module.padding[0] + module.stride[0]
                seq_len -= module.kernel_size[0] * module.dilation[0]
                seq_len = seq_len // module.stride[0]
        return seq_len

class SeqFeedForward(CellModule):
    """
    Applies linear layers to every sequence position,
    equivalent to repeated convolution with kernel_size=1.

    Args:
        hidden_layers: number of intermediate linear layers
        hidden_sizes: custom intermediate layer out_channels
        activation_fn: non-linear activation
        activation_on_last: set `True` to apply activation after last linear layer
    """
    def __init__(self, in_channels, out_channels, hidden_layers=0,
                hidden_sizes=None, bias=True, activation_fn=None, activation_on_last=False):
        # without activation this is linear transform, so extra layers are irrelevant
        if activation_fn is None:
            hidden_layers = 0
        if hidden_sizes is None:
            larger = max(in_channels, out_channels)
            hidden_sizes = [in_channels] + [larger] * hidden_layers + [out_channels]
        else:
            assert len(hidden_sizes) == hidden_layers
            hidden_sizes = [in_channels] + hidden_sizes + [out_channels]

        layers = []
        for i, j in zip(hidden_sizes, hidden_sizes[1:]):
            layers += [nn.Conv1d(i, j, kernel_size=1, bias=bias)]
            if not (activation_fn is None):
                layers += [activation_fn()]
        if (not (activation_fn is None)) and (not activation_on_last):
            layers = layers[:-1]
        super().__init__(layers, activation_fn)


class DilateConvCell(CellModule):

    def __init__(self, in_channels, out_channels, kernel_size,
                hidden_layers=0, dilation=1, dropout_rate=0.,
                activation_fn=nn.ReLU, norm_layer=nn.BatchNorm1d, dropout_layer=nn.Dropout):
        layers = [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation),
                activation_fn()]
        for _ in range(hidden_layers):
            layers += [nn.Conv1d(out_channels, out_channels, kernel_size),
                    activation_fn()]
        if not (norm_layer is None):
            layers += [norm_layer(out_channels)]
        if dropout_rate > 0.:
            layers += [dropout_layer(dropout_rate, inplace=False)]
        super().__init__(layers, activation_fn)


class DilateConvEncoder(CellModule):

    def __init__(self, in_channels, kernel_size, do_pad=True,
                n_cells=1, channel_growth_rate=1.,
                hidden_layers_per_cell=0, dilation=1, dropout_rate=0.,
                activation_fn=nn.ReLU, norm_layer=nn.BatchNorm1d, dropout_layer=nn.Dropout):
        for i, j in zip(channels, channels[1:]):
            layers += [DilateConvCell(channels[i], channels[j], kernel_size,
                        hidden_layers_per_cell, dilation, dropout_rate,
                        activation_fn, norm_layer, dropout_layer)]
        super().__init__(layers, activation_fn)