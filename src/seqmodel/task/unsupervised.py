import sys
sys.path.append('./src')
import torch
import torch.nn as nn

from seqmodel.seq.transform import N_BASE, LambdaModule, one_hot
from seqmodel.task.task import GenericTask


class PredictMaskedToken(GenericTask):

    """
    Task for masked token prediction (Cloze) from BERT.
    Takes in batched sequences of indexes, masks sequence, and calculates loss
    relative to original sequence.

    Args:
        no_loss_prop: (implicit) proportion of data that isn't used to calculate loss
        mask_prop: proportion of data masked in input
        random_prop: proportion of data which is randomly shuffled in input
        keep_prop: proportion of data that is part of loss but unchanged
        mask_value: set to 0 for input to regular layers, -inf for input to softmax (transformer networks)
        n_classes: how many indices to draw from for random replacement
    """
    def __init__(self, encoder, decoder, loss_fn,
                keep_prop=0., mask_prop=0., random_prop=0.,
                mask_value=0, n_classes=N_BASE, allow_nan_loss=False):
        self.set_mask_props(keep_prop, mask_prop, random_prop)
        preprocess = LambdaModule(self.generate_mask_and_permute, self.randomize_input,
                                    one_hot, self.mask_input)
        super().__init__(encoder, decoder, loss_fn, preprocess=preprocess)

        self.mask_value = mask_value
        self.n_classes = n_classes
        self.allow_nan_loss = allow_nan_loss
        self._NO_LOSS_INDEX = 0
        self._MASK_INDEX = 1
        self._RANDOM_INDEX = 2
        self._KEEP_INDEX = 3

    def set_mask_props(self, keep_prop=0., mask_prop=0., random_prop=0.):
        no_loss_prop = 1. - mask_prop - random_prop - keep_prop
        assert (no_loss_prop >= 0. and mask_prop >= 0. \
                and random_prop >= 0. and keep_prop >= 0.)
        self._mask_cutoff = no_loss_prop
        self._random_cutoff = self._mask_cutoff + mask_prop
        self._keep_cutoff = self._random_cutoff + random_prop

    # generate from index vector size
    def generate_mask_and_permute(self, x):
        prob = torch.rand_like(x, dtype=torch.float32)
        self.mask = (prob > self._mask_cutoff).type(torch.int8)     \
                    + (prob > self._random_cutoff).type(torch.int8) \
                    + (prob > self._keep_cutoff).type(torch.int8)
        del prob
        if not self.allow_nan_loss:
            if torch.sum(self.mask) == 0:  # no item was selected for calculating loss
                self.mask[0, 0] = self._KEEP_INDEX  # unmask the first item
        return x  # need to pass x through preprocess

    # apply to index vector
    def randomize_input(self, x):
        return x.masked_scatter(self.mask == self._RANDOM_INDEX,
                torch.randint_like(x, self.n_classes))

    # apply to one-hot vector
    def mask_input(self, x):
        return x.permute(1, 0, 2).masked_fill(
            (self.mask == self._MASK_INDEX), self.mask_value).permute(1, 0, 2)

    def loss(self, x):
        predicted, latent = self(x)
        target_mask = self.mask != self._NO_LOSS_INDEX
        # need to permute to broadcast mask
        # then permute back after getting correct shape
        predicted = predicted.permute(1, 0, 2).masked_select(
                    target_mask).reshape(self.n_classes, -1).permute(1, 0)
        target = x.masked_select(target_mask)
        return predicted, target, latent, self.loss_fn(predicted, target, latent)


class ReconstructionTask(GenericTask):

    def __init__(self):
        pass

    def loss(self, inputs, targets):
        pass


class PredictNextTokenTask(GenericTask):

    def __init__(self):
        pass

    def loss(self, inputs, targets):
        pass
