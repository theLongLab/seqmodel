import sys
sys.path.append('./src')
import torch.nn as nn


class Transformer():

    def __init__(self, embedding, position_encoder, ):
        self._batch_channel_seq_dims = (2, 0, 1)
        x.permute(self._batch_channel_seq_dims)


class SumPositionEncoder():
    pass


class ConcatPositionEncoder():
    pass


class BatchAbsSinePosition():
    pass


class PairwiseRelativePosition():
    pass