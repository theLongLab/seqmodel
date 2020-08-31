import sys
sys.path.append('./src')
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from seqmodel.functional.transform import one_hot
from seqmodel.seqdata.mapseq import create_test_batch
from seqmodel.model.invariant import *


if __name__ == '__main__':
    n_tries = 1000
    do_print_memory_use = False
    dev = torch.device('cuda')

    if do_print_memory_use:
        n_tries = 1

    def print_memory_use():
        if do_print_memory_use:
            print(torch.cuda.memory_allocated(), torch.cuda.memory_cached(), end=' ')

    def test(fn_list, *args_lists):
        for args in product(*args_lists):
            for fn in fn_list:
                names = []
                for arg in args:
                    if type(arg) is torch.Tensor:
                        names.append(arg.shape)
                    else:
                        names.append(arg)
                print(timeit.timeit(lambda: fn(*args), number=n_tries), fn, *names)

    def convolve(x, layer):
        y = layer(x)
        print_memory_use()
        del y

    # test(
    #     [convolve],
    #     [
    #         one_hot(create_test_batch(5, 100)).to(dev),
    #         one_hot(create_test_batch(50, 1000)).to(dev),
    #     ],
    #     [
    #         # RCIConv1d(4, 40, 4, do_reverse=False, do_complement=False).to(dev),
    #         nn.Conv1d(4, 40, 4).to(dev),
    #         GroupConv1d(4, 40, 4).to(dev),
    #         v2GroupConv1d(4, 40, 4).to(dev),
    #         # # RCIConv1d(4, 40, 4, do_reverse=True, do_complement=True).to(dev),
    #         # # RCIConv1d(4, 160, 4, do_reverse=True, do_complement=True).to(dev),
    #         # nn.Conv1d(4, 400, 40).to(dev),
    #         # RCIConv1d(4, 400, 40, do_reverse=False, do_complement=False).to(dev),
    #         nn.Conv1d(4, 40, 40).to(dev),
    #         GroupConv1d(4, 40, 40).to(dev),
    #         v2GroupConv1d(4, 40, 40).to(dev),
    #         nn.Conv1d(4, 400, 4).to(dev),
    #         GroupConv1d(4, 400, 4).to(dev),
    #         v2GroupConv1d(4, 400, 4).to(dev),
    #         nn.Conv1d(4, 400, 40).to(dev),
    #         GroupConv1d(4, 400, 40).to(dev),
    #         v2GroupConv1d(4, 400, 40).to(dev),
    #         # RCIConv1d(4, 400, 40, do_reverse=True, do_complement=True).to(dev),
    #         # RCIConv1d(4, 1600, 40, do_reverse=True, do_complement=True).to(dev),
    #     ],)
        del y

    def fn1(x):
        return torch.cat(
            [
                x,
                reverse_complement(x),
                reverse(x),
                complement(x),
            ], dim=0)
    
    def fn2(x):
        return x.view(x.shape[0], self.out_channels * 4, -1)

    test(
        [convolve],
        [
            one_hot(create_test_batch(5, 100)).to(dev),
            one_hot(create_test_batch(50, 1000)).to(dev),
        ],
        [
            # RCIConv1d(4, 40, 4, do_reverse=False, do_complement=False).to(dev),
            nn.Conv1d(4, 40, 4).to(dev),
            GroupConv1d(4, 40, 4).to(dev),
            v2GroupConv1d(4, 40, 4).to(dev),
            # # RCIConv1d(4, 40, 4, do_reverse=True, do_complement=True).to(dev),
            # # RCIConv1d(4, 160, 4, do_reverse=True, do_complement=True).to(dev),
            # nn.Conv1d(4, 400, 40).to(dev),
            # RCIConv1d(4, 400, 40, do_reverse=False, do_complement=False).to(dev),
            nn.Conv1d(4, 40, 40).to(dev),
            GroupConv1d(4, 40, 40).to(dev),
            v2GroupConv1d(4, 40, 40).to(dev),
            nn.Conv1d(4, 400, 4).to(dev),
            GroupConv1d(4, 400, 4).to(dev),
            v2GroupConv1d(4, 400, 4).to(dev),
            nn.Conv1d(4, 400, 40).to(dev),
            GroupConv1d(4, 400, 40).to(dev),
            v2GroupConv1d(4, 400, 40).to(dev),
            # RCIConv1d(4, 400, 40, do_reverse=True, do_complement=True).to(dev),
            # RCIConv1d(4, 1600, 40, do_reverse=True, do_complement=True).to(dev),
        ],)