import os
import os.path
import numpy as np
import pandas as pd
from pyfaidx import Fasta

import seqmodel.seqdata.dataset.torchvision_utils as tvu


def decompressed_name(filename):
    if tvu._is_tarxz(filename):
        return filename[:-len('.tar.xz')]
    if tvu._is_tar(filename):
        return filename[:-len('.tar')]
    if tvu._is_targz(filename):
        return filename[:-len('.tar.gz')]
    if tvu._is_tgz(filename):
        return filename[:-len('.tgz')]
    if tvu._is_gzip(filename):
        return filename[:-len('.gz')]
    if tvu._is_zip(filename):
        return filename[:-len('.zip')]
    return filename

class DownloadableDataset():

    _DEFAULT_CACHE_ROOT = '.seqmodel/.seqdata/.datasets/.cache'

    def __init__(self, url_root, cache_root=None, download=False, remove_compressed=False):
        self.url_root = url_root
        self._cache_root = os.path.expanduser(cache_root)
        if cache_root is None:
            self.cache_root = os.path.join(os.getcwd(), self._DEFAULT_CACHE_ROOT)
        self.download = download
        self.remove_compressed = remove_compressed

    @property
    def root_name(self):
        raise NotImplementedError('Need to set name for unique cache location')

    @property
    def cache_root(self) -> str:
        return os.path.join(self._cache_root, self.__class__.__name__, self.root_name)

    def cache_path(self, filename):
        return os.path.join(self.cache_root, filename)

    def _check_exists(self, filename):
        return os.path.exists(self.cache_path(filename))

    def validate(self, filename, md5):
        return tvu.check_integrity(self.cache_path(filename), md5)

    def retrieve_or_download(self, filename, md5, decompress=True):
        if decompress:
            # check for decompressed file
            extracted_filename = decompressed_name(filename)
            if not self._check_exists(extracted_filename):
                # if not exists, check for compressed file
                path_to_compressed = self.retrieve_or_download(filename, md5, decompress=False)
                # decompress file and return
                tvu.extract_archive(path_to_compressed, remove_finished=self.remove_compressed)
            if self._check_exists(extracted_filename):
                return self.cache_path(extracted_filename)
            else:
                raise FileNotFoundError('Unable to retrieve or download ' + filename)
        else:
            # check for file
            if not self._check_exists(filename) and self.download:
                # if not exists, download and return
                tvu.download_url(self.url_root + '/' + filename, root=self.cache_root,
                                filename=filename, md5=md5)
            if self._check_exists(filename):
                return self.cache_path(filename)
            else:
                raise FileNotFoundError('Unable to retrieve or download ' + filename)


class FastaSequence():

    def __init__(self, filename, as_raw=True):
        self.fasta = Fasta(filename, as_raw=as_raw)

    def all_intervals(self):
        return SeqIntervals.from_cols(
            list(self.fasta.keys()),
            [0] * len(self.fasta.keys()),
            [len(x) for x in self.fasta.values()])

    @property
    def gap_intervals(self):
        raise NotImplementedError()  # TODO need to incorporate gap detector like contig.py

    def ungap(self):
        return self.all_intervals.remove(self.gap_intervals)


class SeqIntervals():

    _DEFAULT_COL_NAMES = ['names', 'start', 'end']

    def __init__(self, annotation_table, seq_start_end_cols=[0, 1, 2]):
        self.table = annotation_table
        self.seq_start_end_cols = [None] * 3
        for i, col in enumerate(seq_start_end_cols):
            if type(i) is int:
                self.seq_start_end_cols[i] = annotation_table.columns[col]
            else:
                assert col in annotation_table.columns
                self.seq_start_end_cols[i] = col

    @classmethod
    def from_cols(cls, seqs, starts, ends):
        data =  {cls._DEFAULT_COL_NAMES[0]: seqs,
                cls._DEFAULT_COL_NAMES[1]: starts,
                cls._DEFAULT_COL_NAMES[2]: ends}
        return cls(pd.DataFrame(data=data, columns=cls._DEFAULT_COL_NAMES))

    @classmethod
    def from_bed_file(cls, filename, sep='\t'):
        with open(filename, 'r') as file:
            table = pd.read_csv(file, sep=sep, names=cls._DEFAULT_COL_NAMES)
        return cls(table)

    @property
    def names(self):
        return list(self.table.loc[:, self.seq_start_end_cols[0]])

    @property
    def start(self):
        return list(self.table.loc[:, self.seq_start_end_cols[1]])

    @property
    def end(self):
        return list(self.table.loc[:, self.seq_start_end_cols[2]])

    def union(self, intervals):
        pass  #TODO

    def intersect(self, intervals):
        pass  #TODO

    def remove(self, intervals):
        pass  #TODO
