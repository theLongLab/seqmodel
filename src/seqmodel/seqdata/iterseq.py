import sys
sys.path.append('./src')
from math import log, sqrt, ceil
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import IterableDataset
from seqmodel.functional import do_nothing


def bed_from_file(bed_filename):
    return pd.read_csv(bed_filename, sep='\t', names=['seqname', 'start', 'end'])

def fasta_from_file(fasta_filename):
    return Fasta(fasta_filename, as_raw=True)  # need as_raw=True to return strings


class SequentialData():

    # make new copy of sequence internals for multiple data loader workers
    def instance(self):
        raise NotImplementedError()

    # get sequence data
    def get(self, seqname, coord_start, coord_end):
        raise NotImplementedError()

    @property
    def seqnames(self):
        raise NotImplementedError()

    @property
    def starts(self):
        raise NotImplementedError()

    @property
    def ends(self):
        raise NotImplementedError()

    def all_intervals(self):
        return {
            'seqname': self.seqnames,
            'start': self.starts,
            'end': self.ends,
        }


class FastaFile(SequentialData):

    def __init__(self, filename):
        self.filename = filename
        self.instance()
    
    def instance(self):
        self.fasta = Fasta(self.filename, as_raw=True)
    
    def get(self, seqname, coord_start, coord_end):
        return self.fasta[seqname][coord_start:coord_end]

    @property
    def seqnames(self):
        return list(self.fasta.keys())

    @property
    def starts(self):
        return [0] * len(self.fasta.keys())

    @property
    def ends(self):
        return [len(seq) for seq in self.fasta.values()]


class StridedSequence(IterableDataset):

    """
        sequence_data: any object implementing SequentialData interface
        seq_len: length of sequence to return
        include_intervals: sequence intervals to sample from, in the form
            `{'seqname': [list<str>], 'start': [list<int>], 'end': [list<int>]}`.
            If `include_intervals=None`, sample from all sequence data.
        transform: function applied to sequence output (type depends on SequentialData object)
        label_transform: function applied to tuple of (key, coord) for sequence
        sequential: if `True`, this is equivalent to setting `stride=1` and `start_offset=0`
        stride: how many indices to move between samples. Defaults to nearest power of 2
            to square root of total sequence positions
        start_offset: first index to sample from. Defaults to random value.
        sample_freq: how often to sample. Defaults to 1 (every position is start of a sample).
            To have non-overlapping samples, set equal to `seq_len`.
        min_len: controls length of the last sample in each interval.
            If `None`, only return sequences of `seq_len`,
            Otherwise, return sequences of length up to and including `min_len`.
    """
    def __init__(self,
                sequence_data,
                seq_len,
                include_intervals=None,
                transform=torch.nn.Identity(),
                label_transform=do_nothing,
                sequential=False,
                stride=None,
                start_offset=None,
                sample_freq=1,
                min_len=None):

        self.sequence_data = sequence_data
        self.seq_len = seq_len
        self.transform = transform
        self.label_transform = label_transform
        self.include_intervals = include_intervals
        self.sample_freq = sample_freq
        if self.include_intervals is None:  # use entire sequence
            self.include_intervals = self.sequence_data.all_intervals()

        self.min_len = min_len
        if min_len is None:
            self.min_len = self.seq_len

        # if length is negative, remove interval (set length to 0)
        n_samples = [max(0, (y - x - self.min_len + self.sample_freq) // self.sample_freq)
                    for x, y in zip(self.include_intervals['start'], self.include_intervals['end'])]
        self.keys = self.include_intervals['seqname']
        self.coord_offsets = list(self.include_intervals['start'])
        self.last_indexes = np.cumsum(n_samples)
        self.n_seq = np.sum(n_samples)    # number of sample indices
        self._iter_start = 0            # worker start index
        self._iter_end = self.n_seq     # worker end index

        if sequential:  # return sequences in order from beginning
            self.stride = 1
            self.start_offset = 0
        else:
            if stride is None:
                # make sure total positions is odd (this guarantees stride covers all positions)
                if self.n_seq % 2 == 0:
                    self.n_seq -= 1
                # nearest power of 2 to square root of self.n_seq, this gives reasonably spaced positions
                self.stride = 2 ** int(round(log(sqrt(self.n_seq), 2)))
            else:
                self.stride = stride
            # randomly assign start position
            if start_offset is None:
                self.start_offset = torch.randint(self.n_seq, [1]).item()
            else:
                self.start_offset = start_offset

    @staticmethod  # partition total indices among workers
    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sequence_data.instance()  # need new object for concurrency
        n_per_worker = int(ceil(dataset.n_seq / worker_info.num_workers))
        dataset._iter_start = n_per_worker * worker_info.id  # split indexes by worker id
        dataset._iter_end = min(dataset._iter_start + n_per_worker, dataset.n_seq)

    def get_data_loader(self, batch_size, num_workers, collate_fn=None):
        return torch.utils.data.DataLoader(self,
                batch_size=batch_size,
                shuffle=False,              # load sequentially
                num_workers=num_workers,
                collate_fn=collate_fn,      # optional function to preprocess batch
                pin_memory=True,  # pinned memory transfers faster to CUDA
                worker_init_fn=self._worker_init_fn,  # initialize multithread workers
                    # prefetch one batch, this doesn't work for torch < 1.7
                # prefetch_factor=(batch_size // num_workers),
            )

    def index_to_coord(self, i):
        index = (i * self.stride + self.start_offset) % self.n_seq
        # look for last (right side) matching value to skip over any removed (zero length) intervals
        row = np.searchsorted(self.last_indexes, index, side='right')
        # look up sequence name and genomic coordinate from interval table
        key = self.keys[row]
        if row > 0:  # need to find index relative to start of interval
            index -= self.last_indexes[row - 1]
        coord =  self.coord_offsets[row] + (index * self.sample_freq)  # convert to coord
        return key, coord

    def __iter__(self):
        for i in range(self._iter_start, self._iter_end):
            key, coord = self.index_to_coord(i)
            seq = self.sequence_data.get(key, coord, coord + self.seq_len)
            yield self.transform(seq), self.label_transform(key, coord)
