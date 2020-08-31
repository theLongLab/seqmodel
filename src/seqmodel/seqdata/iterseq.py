import sys
sys.path.append('./src')
from math import log, sqrt
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import IterableDataset

from seqmodel.functional.transform import bioseq_to_index


def bed_from_file(bed_filename):
    return pd.read_csv(bed_filename, sep='\t', names=['chr', 'start', 'end'])

def fasta_from_file(fasta_filename):
    return Fasta(fasta_filename, as_raw=True)  # need as_raw=True to return strings


class StridedSequence(IterableDataset):

    def __init__(self, pyfaidx_fasta, seq_len, include_intervals=None,
                sequential=False, stride=0, start_offset=-1):
        self.fasta = pyfaidx_fasta
        self.seq_len = seq_len
        self._cutoff = self.seq_len - 1

        if include_intervals is None:  # use entire fasta sequence
            lengths = [len(seq) - self._cutoff for seq in self.fasta.values()]
            self.keys = list(self.fasta.keys())
            self.coord_offsets = [0] * len(self.fasta.keys())
        else:  # make table of intervals
            # if length is negative, remove interval (set length to 0)
            lengths = [max(0, y - x - self._cutoff)
                        for x, y in zip(include_intervals['start'], include_intervals['end'])]
            self.keys = include_intervals['chr']
            self.coord_offsets = list(include_intervals['start'])
        self.n_seq = np.sum(lengths)
        self.last_indexes = np.cumsum(lengths)

        if sequential:  # return sequences in order from beginning
            self.stride = 1
            self.start_offset = 0
        else:
            if stride is None:
                # make sure total positions is odd (this guarantees stride covers all positions)
                if self.n_seq % 2 == 0:
                    self.n_seq -= 1
                # nearest power of 2 to square root of self.n_seq, this gives nicely spaced positions
                self.stride = 2 ** int(round(log(sqrt(self.n_seq), 2)))
            else:
                self.stride = stride
            # randomly assign start position (this will be different for each dataloader worker)
            if start_offset is None:
                self.start_offset = torch.randint(self.n_seq, [1]).item()
            else:
                self.start_offset = start_offset

    @classmethod
    def from_file(cls, fasta_filename, seq_len, include_intervals=None,
                sequential=False, stride=0, start_offset=-1):
        fasta = fasta_from_file(fasta_filename)  # need as_raw=True to return strings
        return cls(fasta, seq_len, include_intervals, sequential, stride, start_offset)

    def index_to_coord(self, i):
        index = (i * self.stride + self.start_offset) % self.n_seq
        # look for last (right side) matching value to skip over any removed (zero length) intervals
        row = np.searchsorted(self.last_indexes, index, side='right')
        # look up sequence name and genomic coordinate from interval table
        key = self.keys[row]
        if row == 0:  # need to find index relative to start of interval
            index_offset = 0
        else:
            index_offset = self.last_indexes[row - 1]
        coord =  self.coord_offsets[row] + index - index_offset
        return key, coord

    def __iter__(self):
        for i in range(self.n_seq):
            key, coord = self.index_to_coord(i)
            seq = self.fasta[key][coord:coord + self.seq_len]
            yield bioseq_to_index(seq)
