import torch
import numpy as np
from math import log, sqrt
from torch.utils.data import IterableDataset


# set batch_size=None in data loader
class IterSequence(IterableDataset):

    def __init__(self, fasta, seq_len, batch_size, included_intervals=None, sequential=True):
        self.fasta = fasta
        self.seq_len = seq_len
        self._cutoff = self.seq_len - 1
        self.batch_size = batch_size

        if included_intervals is None:  # use entire fasta sequence
            lengths = [len(seq) - self._cutoff for seq in self.fasta.values()]
            self.keys = list(self.fasta.keys())
            self.coord_offsets = [0] * len(self.fasta.keys())
        else:  # make table of intervals
            # if length is negative, remove interval (set length to 0)
            lengths = [max(0, y - x - self._cutoff) for x, y in zip(included_intervals['start'], included_intervals['end'])]
            self.keys = included_intervals['chr']
            self.coord_offsets = list(included_intervals['start'])
        self.n_seq = np.sum(lengths)
        self.last_indexes = np.cumsum(lengths)

        if sequential:  # return sequences in order from beginning
            self.stride = 1
            self.start_offset = 0
        else:
            # make sure total positions is odd (this guarantees stride covers all positions)
            if self.n_seq % 2 == 0:
                self.n_seq -= 1
            # nearest power of 2 to square root of self.n_seq, this gives nicely spaced positions
            self.stride = 2 ** int(round(log(sqrt(self.n_seq), 2)))
            # randomly assign start position (this will be different for each dataloader worker)
            self.start_offset = torch.randint(self.n_seq, [1]).item()

    def index_to_coord(self, i):
        index = (i * self.stride + self.start_offset) % self.n_seq
        # look for last (right side) matching value to skip over any removed (zero length) intervals
        row = np.searchsorted(self.last_indexes, index, side='right')
        # look up sequence name and genomic coordinate from interval table
        key = self.keys[row]
        coord = index - self.last_indexes[row] - self._cutoff + self.coord_offsets[row]
        return key, coord

    def __iter__(self):
        for i in range(self.n_seq):
            key, coord = self.index_to_coord(i)
            if coord + self.seq_len == 0:
                yield self.fasta[key][coord:]
            else:
                yield self.fasta[key][coord:coord + self.seq_len]
