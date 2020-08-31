import sys
sys.path.append('./src')
import torch
import torch.nn as nn

from seqmodel.seq.transform import N_BASE, Compose, one_hot


class PositionMask():

    """
    Task for masked token prediction (Cloze) from BERT.
    Takes in batched sequences of indexes, masks sequence, and calculates loss
    relative to original sequence.

    All functions assume input has dimensions (batch, seq_len) or (batch, channels, seq_len)
    This follows the standard for convolutions in pytorch, not for transformers.

    Args:
        no_loss_prop: (implicit) proportion of data that isn't used to calculate loss
        mask_prop: proportion of data masked in input
        random_prop: proportion of data which is randomly shuffled in input
        keep_prop: proportion of data that is part of loss but unchanged
        n_classes: how many indices to draw from for random replacement
    """
    _NO_LOSS_INDEX = 0
    _MASK_INDEX = 1
    _RANDOM_INDEX = 2
    _KEEP_INDEX = 3

    def __init__(self, mask_prop=0., random_prop=0., keep_prop=0., n_classes=N_BASE):
        self.n_classes = n_classes
        self.set_mask_props(mask_prop, random_prop, keep_prop)

    def set_mask_props(self, mask_prop=0., random_prop=0.,keep_prop=0.):
        no_loss_prop = 1. - (mask_prop + random_prop + keep_prop)
        assert (no_loss_prop >= 0. and mask_prop >= 0. \
                and random_prop >= 0. and keep_prop >= 0.)
        self._mask_cutoff = no_loss_prop
        self._random_cutoff = self._mask_cutoff + mask_prop
        self._keep_cutoff = self._random_cutoff + random_prop

    # generate from index vector size
    def generate(self, x, require_loss_pos=True):
        prob = torch.rand_like(x, dtype=torch.float32)
        self.mask_val = (prob > self._mask_cutoff).type(torch.int8) \
                    + (prob > self._random_cutoff).type(torch.int8) \
                    + (prob > self._keep_cutoff).type(torch.int8)
        del prob
        if require_loss_pos:
            if torch.sum(self.mask_val) == 0:  # no item was selected for calculating loss
                self.mask_val[0, 0] = self._KEEP_INDEX  # unmask the first item
        return self.mask_val

    # apply to index vector
    def randomize_input(self, x):
        return x.masked_scatter(self.mask_val == self._RANDOM_INDEX,
                torch.randint_like(x, self.n_classes))

    # apply to one-hot vector
    def mask_fill(self, x, fill_value=0):
        if x.dim() == 3:
            return x.permute(1, 0, 2).masked_fill(
                    (self.mask_val == self._MASK_INDEX), fill_value).permute(1, 0, 2)
        else:
            return x.masked_fill((self.mask_val == self._MASK_INDEX), fill_value)
    
    def get(self, mask_value=float('-inf')):
        if type(mask_value) is bool:
            if mask_value:
                return self.mask_val == self._MASK_INDEX
            return self.mask_val != self._MASK_INDEX
        mask = torch.zeros_like(self.mask_val, dtype=torch.float)
        return mask.masked_fill((self.mask_val == self._MASK_INDEX), mask_value)

    def attn_mask(self, x, mask_value=float('-inf'), generate_new_mask=True,
                        randomize_input=True, mask_fill=False):
        if generate_new_mask:
            self.generate(x)
        if mask_fill:
            x = self.mask_fill(x)
        if randomize_input:
            x = self.randomize_input(x)
        return x, self.get(mask_value=mask_value)

    # order of dimensions is consistent between outputs but not consistent from input to output
    # this is intended to feed into loss function or other aggregation function
    def select(self, *xargs):
        target_mask = self.mask_val != self._NO_LOSS_INDEX
        output = []
        for x in xargs:
            # need to permute to broadcast mask
            # then permute back after getting correct shape
            if x.dim() == 3:
                output.append(x.permute(1, 0, 2).masked_select(
                        target_mask).reshape(self.n_classes, -1).permute(1, 0))
            else:
                output.append(x.masked_select(target_mask))
        return output


class NextTokenMask():

    def __init__(self):
        pass

