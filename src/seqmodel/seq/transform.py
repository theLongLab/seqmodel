import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq


BASE_TO_INDEX = {
    'A': 0, 'a': 0,
    'G': 1, 'g': 1,
    'C': 2, 'c': 2,
    'T': 3, 't': 3,
    'N': 4, 'n': 4,
    }
EMPTY_INDEX = 4
INDEX_TO_BASE = ['A', 'G', 'C', 'T']
N_BASE = 4


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


class LambdaModule(nn.Module):

    def __init__(self, *fn_list):
        super().__init__()
        self.fn_list = fn_list

    def forward(self, x):
        for fn in self.fn_list:
            x = fn(x)
        return x
