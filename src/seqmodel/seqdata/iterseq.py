import sys
sys.path.append('./src')
from math import log, sqrt, ceil
import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
from torch.utils.data import IterableDataset


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
        sequential: if True, this is equivalent to setting stride=1 and start_offset=0
    """
    def __init__(self, sequence_data, seq_len, include_intervals=None,
                transforms=torch.nn.Identity(),
                sequential=False, stride=None, start_offset=None):
        self.sequence_data = sequence_data
        self.seq_len = seq_len
        self._cutoff = self.seq_len - 1
        self.transforms = transforms
        self.include_intervals = include_intervals
        if self.include_intervals is None:  # use entire sequence
            self.include_intervals = self.sequence_data.all_intervals()
        self._gen_interval_table(self.include_intervals)

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

    def _gen_interval_table(self, include_intervals):
        # if length is negative, remove interval (set length to 0)
        lengths = [max(0, y - x - self._cutoff)
                    for x, y in zip(include_intervals['start'], include_intervals['end'])]
        self.keys = include_intervals['seqname']
        self.coord_offsets = list(include_intervals['start'])
        self.n_seq = np.sum(lengths)
        self.last_indexes = np.cumsum(lengths)
        self._iter_start = 0
        self._iter_end = self.n_seq

    @staticmethod
    def _worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.sequence_data.instance()  # need new object for concurrency
        n_per_worker = int(ceil(dataset.n_seq / worker_info.num_workers))
        dataset._iter_start = n_per_worker * worker_info.id  # split indexes by worker id
        dataset._iter_end = min(dataset._iter_start + n_per_worker, dataset.n_seq)

    def get_data_loader(self, batch_size, num_workers):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, worker_init_fn=self._worker_init_fn)

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
        for i in range(self._iter_start, self._iter_end):
            key, coord = self.index_to_coord(i)
            seq = self.sequence_data.get(key, coord, coord + self.seq_len)
            yield self.transforms(seq), (key, coord)
