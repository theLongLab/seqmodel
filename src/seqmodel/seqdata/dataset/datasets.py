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

    _DEFAULT_CACHE_ROOT = '.cached_data/datasets'

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


class FastaDataset():

    def __init__(self, filename, as_raw=True):
        # TODO currently transforms depend on as_raw=True to manipulate str, maybe change transforms?
        self.fasta = Fasta(filename, as_raw=as_raw)

    @property
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

    """
    Defines start and endpoints within a set of named sequences.
    Intervals must be unique (by name) and non-overlapping.

    seqname_start_end_index: list of int indices or str names of columns relevant
            SeqIntervals, must be in order [seqname, start, end]
    """
    def __init__(self, annotation_table, seqname_start_end_index=[0, 1, 2], nonzero=True):
        self.table = annotation_table
        self._index = pd.Index([])
        for i, col in enumerate(seqname_start_end_index):
            if type(col) is int:
                self._index = self._index.insert(i, annotation_table.columns[col])
            else:
                assert col in annotation_table.columns
                self._index = self._index.insert(i, col)
        if nonzero:
            self.table = self._filter_by_len()  # get rid of 0 or negative length intervals

    @classmethod
    def from_cols(cls, seqs, starts, ends, nonzero=True):
        data =  {cls._DEFAULT_COL_NAMES[0]: seqs,
                cls._DEFAULT_COL_NAMES[1]: starts,
                cls._DEFAULT_COL_NAMES[2]: ends}
        return cls(pd.DataFrame(data=data, columns=cls._DEFAULT_COL_NAMES), nonzero=nonzero)

    @classmethod
    def from_bed_file(cls, filename, sep='\t', nonzero=True):
        with open(filename, 'r') as file:
            table = pd.read_csv(file, sep=sep, names=cls._DEFAULT_COL_NAMES)
        return cls(table, nonzero=nonzero)

    @property
    def seqnames(self):
        return self.table[self._index[0]]

    @property
    def start(self):
        return self.table.loc[:, self._index[1]]

    @property
    def end(self):
        return self.table.loc[:, self._index[2]]

    @property
    def length(self):
        return self.end - self.start
    
    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return repr(self.table)

    def __str__(self):
        return repr(self.table)

    def clone(self, new_table=None, deep_copy=False):
        table = new_table
        if new_table is None:
            table = self.table
        if deep_copy:
            table = table.copy()
        return SeqIntervals(table, self._index, nonzero=False)

    def filter(self, *loc_labels, col_to_search=None):
        if col_to_search is None:
            col_to_search = self.seqnames
        return self.clone(self.table[col_to_search.isin(loc_labels)], False)

    def _filter_by_len(self, min_len=1, max_len=None):
        if max_len is None:
            return self.table.loc[self.length >= min_len]
        else:  # use element-wise and `&`
            return self.table.loc[(self.length >= min_len) & (self.length <= max_len)]

    # max_len is inclusive
    def filter_by_len(self, min_len=1, max_len=None):
        return self.clone(self._filter_by_len(min_len, max_len), False)

    # warning: this is an in place operation!
    def _merge_overlaps(self, min_allowable_gap):
        self.table = self.table.sort_values([self._index[0], self._index[1]])
        distance_to_next_interval = self.start[1:].values - self.end[:-1].values
        is_same_sequence = self.seqnames[1:].values == self.seqnames[:-1].values
        # merge_candidates guaranteed to contain the first of multiple overlapping positions
        # but not necessarily all of the subsequent positions: counter example is
        # 0-10, 1-2, 3-4, merge_candidates will be [True, False] even though 3-4 overlaps 0-10
        # this gives a starting point for iterating over intervals sequentially, which is slow
        merge_candidates = (distance_to_next_interval < min_allowable_gap) & is_same_sequence

        rows_to_remove = []
        end_col = self.table.columns.get_loc(self._index[2])
        for i in np.nonzero(merge_candidates)[0]:
            j = i + 1
            # check if merge_candidates are between the same named sequence
            while j < len(self.table) and is_same_sequence[j - 1] \
                    and self.start.iloc[j] - self.end.iloc[i] < min_allowable_gap:
                # if true, merge the intervals and the next ones
                # until the distance to next is greater than min_allowable_gap
                # extend first interval
                # note: cannot update array values using chained indexing, hence use `.loc`
                # see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
                self.table.iloc[i, end_col] = max(self.end.iloc[i], self.end.iloc[j])
                rows_to_remove.append(j)  # remove subsequent intervals
                j += 1
        self.table = self.table.drop(self.table.iloc[rows_to_remove].index)
        return self

    # reimplement this for labelled intervals
    def columns_match(self, interval):
        return (self.table.columns == interval.table.columns).all() \
                and (self._index == interval._index).all()

    def _append_tables(self, *intervals):
        new_table = self.table
        for i in intervals:
            if self.columns_match(i):
                new_table = new_table.append(i.table)
            else:
                raise ValueError('SeqInterval object columns do not match ', str(interval))
        return new_table

    """
    Set union (addition) of current SeqIntervals object with other SeqIntervals

        in_place: default `False` creates a new copy of underlying table data.
            If `in_place=True`, modify calling object's table.
    """
    def union(self, *intervals, min_allowable_gap=1, in_place=False):
        table = self._append_tables(*intervals)
        if in_place:
            self.table = table
            return self._merge_overlaps(min_allowable_gap)
        else:
            interval_obj = self.clone(table, True)
            return interval_obj._merge_overlaps(min_allowable_gap)

    def intersect(self, *intervals, in_place=False):
        pass  #TODO

    def remove(self, *intervals, in_place=False):
        pass  #TODO
