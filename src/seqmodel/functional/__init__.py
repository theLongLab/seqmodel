import sys
sys.path.append('./src')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq

from seqmodel import BASE_TO_INDEX, EMPTY_INDEX, INDEX_TO_BASE, N_BASE


# only works on one sequence at a time (1 dimension)
def bioseq_to_index(bioseq):
    str_array = np.array(list(bioseq))
    int_array = np.empty(len(bioseq), dtype='int8')
    for base, index in BASE_TO_INDEX.items():
        match = np.char.equal(str_array, base)
        int_array[match] = index
    return torch.LongTensor(int_array)

# only works on one sequence at a time (1 dimension)
def index_to_bioseq(tensor):
    assert len(tensor.shape) == 1
    return Seq(''.join([INDEX_TO_BASE[i] for i in tensor.cpu().detach().numpy()]))

# append sequences start_flank, end_flank to start and end of seq
# assumes index tensors
def flank(seq, start_flank=None, end_flank=None):
    if start_flank is None:
        start_flank = torch.tensor([], dtype=seq.dtype)
    if end_flank is None:
        end_flank = torch.tensor([], dtype=seq.dtype)
    return torch.cat([start_flank, seq, end_flank], dim=0)

#assumes batch x length sequence
# shuffles prop*batch_n sequences, guarantees they are mapped to new sequences
# also returns vector indicating which ones are shuffled
def permute(batch, prop=1.):
    with torch.no_grad():
        n_batch = batch.shape[0]
        n_perm = int(n_batch * prop)

        cycle_x = torch.randperm(n_batch)[:n_perm]  # indexes to draw items from
        cycle_y = torch.cat((cycle_x[1:], cycle_x[0:1]), dim=0)  # indexes to place items
        select_indexes = torch.arange(n_batch)
        select_indexes[cycle_y] = cycle_x  # apply permutation on indexes, then use indexed select
        select_indexes = select_indexes.to(batch.device)

        # generate indicator vector
        is_permuted = torch.zeros_like(select_indexes, dtype=torch.bool)
        is_permuted[cycle_x] = True
        return is_permuted, batch.index_select(0, select_indexes)

# random subsequence between min_len and max_len, using last dim
# if len is less than max_len, crop between min_len and len
# if either of crop_start or crop_end are False, only crop from one side
def random_crop(sequence, min_len, max_len, crop_start=True, crop_end=True):
    assert min_len >= 0 and max_len >= min_len and sequence.size(-1) >= min_len
    src_len = sequence.size(-1)
    # target up to and including max_len or seq_len
    tgt_len = min(src_len, torch.randint(min_len, max_len + 1, [1]).item())
    if crop_start and crop_end:
        offset = torch.randint(0, src_len - tgt_len + 1, [1]).item()
        return sequence[..., offset:offset+tgt_len]
    elif crop_start:
        return sequence[..., -tgt_len:]
    elif crop_end:
        return sequence[..., :tgt_len]
    else:
        return sequence

# randomly position source seq in target, cut source to size if needed
# in place operation
def random_seq_fill(source, target):
    src_len = source.size(-1)
    tgt_len = target.size(-1)
    offset = torch.randint(0, abs(src_len - tgt_len) + 1, [1]).item()
    if src_len > tgt_len:
        target = source[..., offset:offset + tgt_len]
    else:
        target[..., offset:offset + src_len] = source
    return target

def one_hot(index_sequence, indexes=range(N_BASE), dim=1):
    with torch.no_grad():
        return torch.stack([(index_sequence == i).float() for i in indexes], dim=dim)

def one_hot_to_index(x):
    return torch.argmax(x, dim=1)

def softmax_to_index(tensor, threshold_score=float('-inf'),
                    no_prediction_index=EMPTY_INDEX):
    values, indexes = torch.max(tensor, dim=1)
    return indexes.masked_fill((values < threshold_score), no_prediction_index)

# below functions for one hot tensors
def complement(x):
    return torch.flip(x, [1])

def reverse(x):
    return torch.flip(x, [2])

def reverse_complement(x):
    return torch.flip(x, [1, 2])

def swap(x, dim=1):
    a, b = torch.split(x, x.shape[dim] // 2, dim=dim)
    return torch.cat([b, a], dim=dim)

def do_nothing(*args):
    return args

# composes multiple functions together
class Compose(nn.Module):

    def __init__(self, *fn_list):
        super().__init__()
        self.fn_list = fn_list

    def forward(self, x):
        for fn in self.fn_list:
            x = fn(x)
        return x
