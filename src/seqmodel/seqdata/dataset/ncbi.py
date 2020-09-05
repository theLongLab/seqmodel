import sys
sys.path.append('./src')
import math
import warnings
import numpy as np
import pandas as pd
from pyfaidx import Fasta

from seqmodel.seqdata.dataset.datasets import *


class NCBIDataset(DownloadableDataset, FastaDataset):

    """
    Download reference assemblies from Genome Research Consortium, NCBI.
    All file paths are the download URLs (e.g. `*_genomic.fna.gz`).
    To get locally cached, decompressed files, use `md5_retrieve_or_download()`
    in `utils.py`.

    """
    _NCBI_FTP_ROOT = 'https://ftp.ncbi.nlm.nih.gov/genomes/all'
    _NAMES_AND_ACCN = np.array([
        #TODO generate automatically from database or web scrape
        # name, GenBank accession, md5 checksum of md5checksums.txt
            ['GRCh37.p13', 'GCA_000001405.14', ''], #TODO add md5 sums
            ['GRCh38.p13', 'GCA_000001405.28', 'b31b1174e6aa6213e08b94dc4d130779'],
            ['ASM308665v1', 'GCA_003086655.1', 'a39d8a1e20172b7dd0703c125ba27b19'],
        ])
    _FASTA_SUFFIX = '_genomic.fna.gz'
    _FAIDX_SUFFIX = '_genomic.fna.fai'
    _FASTA_INDEX_HEADER = [
        'contig_name', 'n_bases', 'byte_index', 'bases_per_line', 'bytes_per_line'
        ]
    _GAP_SUFFIX = '_genomic_gaps.txt'
    _GAP_HEADER = ['accession.version', 'start', 'stop', 'gap_length', 'gap_type', 'linkage_evidence']
    _REPORT_SUFFIX = '_assembly_report.txt'
    _REPORT_HEADER = ['Sequence-Name', 'Sequence-Role', 'Assigned-Molecule',
                    'Assigned-Molecule-Location/Type', 'GenBank-Accn', 'Relationship',
                    'RefSeq-Accn', 'Assembly-Unit', 'Sequence-Length', 'UCSC-style-name']
    _REGION_SUFFIX = '_assembly_region.txt'
    _REGION_HEADER = ['Region-Name', 'Chromosome', 'Chromosome-Start',
                    'Chromosome-Stop', 'Scaffold-Role', 'Scaffold-GenBank-Accn',
                    'Scaffold-RefSeq-Accn', 'Assembly-Unit']
    _CHECKSUM_HEADER = ['md5', 'path']
    _CHECKSUM_FILE = 'md5checksums.txt'

    def __init__(self, url_root, checksumfile_md5,
                cache_root=None, download=False, remove_compressed=False):
        # note: does not call super().__init__ for FastaDataset since that is a wrapper
        # for Fasta objects, objects need to be handled differently in DownloadableDataset
        super().__init__(url_root, cache_root, download, remove_compressed)
        self.checksumfile_md5 = checksumfile_md5
        self._checksums = None
        self._gap_table = None
        self._report_table = None
        self._region_table = None
        self._fasta = None

    @classmethod
    def from_name(cls, name_or_accn, cache_root=None, download=False, remove_compressed=False):
        rows, _ = np.where(cls._NAMES_AND_ACCN == name_or_accn)
        name, accn, checksum = cls._NAMES_AND_ACCN[rows[0]]
        return cls('/'.join([cls._NCBI_FTP_ROOT, accn[:3], accn[4:7],
                    accn[7:10], accn[10:13], accn + '_' + name]),
            checksum, cache_root, download, remove_compressed)

    @property
    def root_name(self):
        return self.url_root.rpartition('/')[-1]

    @property
    def fasta_file(self):
        filename = self.root_name + self._FASTA_SUFFIX
        return self.md5_retrieve_or_download(filename)

    @property
    def gap_table(self):
        if self._gap_table is None:
            self._gap_table = self._get_table(self._GAP_SUFFIX, self._GAP_HEADER)
        return self._gap_table

    @property
    def gap_intervals(self):
        return SeqIntervals(self.gap_table, seq_start_end_cols=self._GAP_HEADER[0:2])

    @property
    def report_table(self):
        if self._report_table is None:
            self._report_table = self._get_table(self._REPORT_SUFFIX, self._REPORT_HEADER)
        return self._report_table

    @property
    def region_table(self):
        if self._region_table is None:
            self._region_table = self._get_table(self._REGION_SUFFIX, self._REGION_HEADER)
        return self._region_table

    @property
    def fasta(self, force_rebuild=False):
        if self._fasta is None:
            faidx_file = self.root_name + self._FAIDX_SUFFIX
            rebuild = force_rebuild  # generate the index if missing, or if force_rebuild=True
            if force_rebuild or not self._check_exists(faidx_file):
                try:  # try generating index from seq length data in assembly report
                    self.generate_faidx_from_assembly_report()
                    rebuild = False
                except:  # let Fasta generate the index by iterating through the sequence directly
                    warnings.warn('Failed to build `*_genomic.fai` from `*_assembly_region.txt`, \
                                iterating through fasta file directly instead...')
                    rebuild = True 
            self._fasta = Fasta(self.fasta_file, rebuild=rebuild)
        return self._fasta  # TODO: Fasta keeps file handle open until explicitly closed with close(), need to expose this

    def _get_table(self, suffix, header=None):
        filename = self.root_name + suffix
        try:
            with open(self.md5_retrieve_or_download(filename), 'r') as file:
                line_start, line, prev_line =  0, '#', ''
                while line[0] == '#':  # strip all comment lines starting with #
                    line_start = file.tell()
                    prev_line = line
                    line = file.readline()
                file.seek(line_start)  # move file pointer back to start of first line without #
                if header is None:  # use last comment line as header
                    header = prev_line[1:].rstrip().split('\t')
                table = pd.read_csv(file, sep='\t', names=header)
        except:  # unable to open file, return empty table
            warnings.warn("Unable to download table {} from {}".format(filename, self.url_root))
            table = pd.DataFrame(columns=header)
        return table

    def generate_faidx_from_assembly_report(self):
        report = self.get_report_table()
        genbank_accn = report[self._REPORT_HEADER[4]]
        seq_len = report[self._REPORT_HEADER[8]]
        byte_index = 0
        fasta_index = []
        with open(self.fasta_file, 'r') as file:
            while True:
                header = file.readline()
                sample_line = file.readline()
                if header == '':
                    break
            assert header[0] == '>'  # the first character of a header line

            accn = header.partition(' ')[0]
            n_bases = seq_len[np.where(genbank_accn == accn)[0][0]]
            byte_index += len(header)
            bases_per_line = len(sample_line.rstrip())
            bytes_per_line = len(sample_line)

            fasta_index.append('\t'.join(accn, n_bases, byte_index, bases_per_line, bytes_per_line))
            byte_index += int(math.ceil(seq_len / bases_per_line)) * bytes_per_line
            file.seek(byte_index)
        return fasta_index

    def md5_retrieve_or_download(self, filename, decompress=True):
        if self._checksums is None:
            md5file = super().retrieve_or_download(
                        self._CHECKSUM_FILE, self.checksumfile_md5, decompress=decompress)
            self._checksums = pd.read_csv(md5file, sep='  ', names=self._CHECKSUM_HEADER,
                            engine='python')  # the c parser doesn't work with double space sep
        index = np.where(self._checksums[self._CHECKSUM_HEADER[1]] == './' + filename)[0][0]
        md5 = self._checksums[self._CHECKSUM_HEADER[0]][index]
        return super().retrieve_or_download(filename, md5, decompress=decompress)

    def translate_accn(name_or_accn, accn_type=None):
        ACCN_HEADERS = [self._REPORT_HEADER[x] for x in [0, 4, 6, 9]]
        if accn_type is None:
            coords = np.where(self.report_table == name_or_accn)
        else:
            assert accn_type in ACCN_HEADERS
            coords = np.where(self.report_table[accn_type] == name_or_accn)
        if len(coords[0] > 1):
            raise ValueError("name_or_accn is not unique, returning first record retrieved by \
                    numpy.where(). This may be resolved by specifying accn_type as one of \
                    'Sequence-Name', 'GenBank-Accn', 'RefSeq-Accn', or 'UCSC-style-name'. ")
        return tuple(self.report_table.iloc[coords[0][0]][ACCN_HEADERS])
