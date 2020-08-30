import sys
sys.path.append('./src')
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, n_dims, n_heads, n_layers, , feedforward_dims=2048, dropout=0.1):
        self._batch_channel_seq_dims = (2, 0, 1)
        self.embedding = embedding
        self.position_encoder = position_encoder
        self.transformer = nn.Transformer(self.hparams.n_dims, self.hparams.n_heads, self.hparams.n_layers,
                    0, self.hparams.feedforward_dims, self.hparams.dropout,
                    custom_encoder=embedding)
    
    def forward(self, x):
        x.permute(self._batch_channel_seq_dims)


class SumPositionEncoder():

    def __init__(self, position_gen)


class ConcatPositionEncoder():
    pass


class SinusoidalPosition():
    pass


class PairwiseRelativePosition():
    pass