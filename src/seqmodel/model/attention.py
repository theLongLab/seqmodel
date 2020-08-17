import sys
sys.path.append('./src')
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, embedding, position_encoder):
        self._batch_channel_seq_dims = (2, 0, 1)
        self.embedding = embedding
        self.position_encoder = position_encoder
        x.permute(self._batch_channel_seq_dims)
        nn.Transformer(d_model, nhead, num_encoder_layers, )


class SumPositionEncoder():
    pass


class ConcatPositionEncoder():
    pass


class BatchAbsSinePosition():
    pass


class PairwiseRelativePosition():
    pass