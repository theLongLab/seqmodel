import sys
sys.path.append('./src')
import torch
import torch.nn as nn

from seqmodel.functional import N_BASE, Compose, one_hot


"""
Functions to assist masked token prediction (Cloze) from BERT.
Takes in batched sequences of indexes, masks sequence, and calculates loss
relative to original sequence.

All functions assume input has dimensions (batch, seq_len) or (batch, channels, seq_len)
This follows the standard for convolutions in pytorch, not for transformers.

Args:
    mask_props: list of n proportions for which indexed values 0...n are assigned
        index of n is for any remaining unmasked values
"""
def prop_cutoffs(mask_props):
    cutoffs = [None] * len(mask_props)
    for i, prop in enumerate(mask_props):
        assert prop >= 0.
        cutoffs[i] = sum(mask_props[:(i + 1)])
    no_loss_prop = 1. - sum(mask_props)
    assert no_loss_prop >= 0.
    return cutoffs

# generate mask matching shape of x
# mask_props indicate proportion of nonzero indexes
# e.g. [0.1, 0.2] gives P(X=0) = 0.7, P(X=1) = 0.1, P(X=2) = 0.2
def generate_mask(x, mask_props, require_loss_pos=False):
    n_indexes = len(mask_props)
    cutoffs = prop_cutoffs(mask_props)
    if n_indexes < 2 ** 8:
        dtype = torch.int8
    else:
        dtype = torch.long
    prob = torch.rand_like(x, dtype=torch.float32)
    mask = torch.zeros_like(x, dtype=dtype)
    for i in range(n_indexes, 0, -1):
        mask = mask.masked_fill(prob < cutoffs[i-1], i)
    del prob
    if require_loss_pos:  # make last item's mask index nonzero to avoid NaN loss
        if torch.sum(mask) == 0:  # no item was selected for calculating loss
            mask[-1, -1] = 1  #(last position avoids conflict with classification token in BERT)
    return mask

# apply to index vector
# random classes must be in {0, 1, ... n_classes - 1}
# randomized class is guaranteed different from original by applying nonzero sum and modulo
def mask_randomize(x, bool_mask, n_classes):
    randomized = torch.remainder(x + torch.randint_like(x, 1, n_classes), n_classes)
    return x.masked_scatter(bool_mask, randomized)

# apply to either one-hot with (batch, channels, length) dims
# or index vector (batch, length)
def mask_fill(x, bool_mask, fill_value):
    # need to permute to broadcast mask, then permute back
    if x.dim() == 3:
        return x.permute(1, 0, 2).masked_fill(
                bool_mask, fill_value).permute(1, 0, 2)
    else:
        return x.masked_fill(bool_mask, fill_value)

# order of dimensions is consistent between outputs but not consistent from input to output
# this is intended to feed into loss function or other aggregation function
def mask_select(x, bool_mask):
    # need to permute to broadcast mask, then permute back and get correct shape
    if x.dim() == 3:
        n_classes = x.shape[1]  # assumes (batch, channels, seq_len)
        return x.permute(1, 0, 2).masked_select(
                bool_mask).reshape(n_classes, -1).permute(1, 0)
    else:
        return x.masked_select(bool_mask)
