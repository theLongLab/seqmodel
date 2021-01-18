from collections import deque
import vcf
import pandas as pd
from Bio import SeqIO

"""
Standalone tools for extracting interval information from genome files.
"""

def map_contigs(fasta_filename, buffer=65536):
    contigs = ByteStreamContigs()
    with open(fasta_filename, 'rb') as file:
        while True:
            data = file.read(buffer)
            if not data:
                contigs.close()
                break
            for byte in data:
                contigs.read(byte)
    return pd.DataFrame(contigs.contigs,
            columns=['header', 'coord_length', 'byte_offset', 'start_coord'])


class ByteStreamContigs():

    # byte values
    _NEWLINE = 10
    _N_LOWER = 110
    _N_UPPER = 78
    _GREATER_THAN = 62

    def __init__(self, is_header=False, is_empty=True, offset=0, coord=0,
                header=b'', start_offset=0, start_coord=0, contigs=[]):
        self.is_header = is_header
        self.is_empty = is_empty
        self.offset = offset  # distance in file
        self.coord = coord  # distance in bases from header
        self.header = header
        self.start_offset = start_offset
        self.start_coord = start_coord
        self.contigs = contigs

    def _reset_coord(self):
        self.is_header = False
        self.is_empty = True
        self.coord = 0
    
    def _mark_contig_start(self):
        self.start_coord = self.coord
        self.start_offset = self.offset

    def _add_contig(self):
        length = self.coord - self.start_coord
        # length in base pairs, offset in bytes from file start, genome coordinate at start of sequence
        self.contigs.append([self.header.decode('utf-8'), length, self.start_offset, self.start_coord])

    def read(self, byte):
        # if header, read until newline, save header, start_coord = 0, is_empty = True
        if self.is_header:
            if byte == self._NEWLINE:
                self._reset_coord()
            else:
                self.header += bytes([byte])
        else:
            if byte == self._GREATER_THAN:
                if not self.is_empty:
                    self._add_contig()
                self.header = b''
                self.is_header = True
            elif byte == self._NEWLINE:
                pass
            else:
                # indicates change from empty to not empty and vice versa
                if (byte == self._N_LOWER or byte == self._N_UPPER) != self.is_empty:
                    # if empty, read until nonempty incrementing start_coord, save start_coord
                    if self.is_empty:
                        self._mark_contig_start()
                    # if nonempty, read until empty incrementing start_coord, save length, save record as header, length, start_coord
                    else:
                        self._add_contig()
                    self.is_empty = not self.is_empty
                self.coord += 1
        self.offset += 1

    # call after last byte of last file is read
    def close(self):
        if not self.is_header and not self.is_empty:
            self._add_contig()


def map_variant_density(vcf_file, window_len, density_threshold, min_gap_between_intervals=1):
    reader = vcf.Reader(filename=vcf_file)
    window = window_len + min_gap_between_intervals
    var_queue = deque()
    intervals = []
    interval_start = -1  # negative if not in interval
    # assume records are sorted (by chrom, pos)
    for record in reader:
        print(record.CHROM, record.POS, end='\r')
        # if moving to different region, close previous interval and reset
        if interval_start > 0 and record.CHROM != var_queue[-1].CHROM:
            intervals.append([var_queue[-1].CHROM, interval_start, var_queue[-1].POS + window_len])
            var_queue = deque()  # empty queue
            interval_start = -1

        cur_pos = record.POS  # move window to next variant position
        # remove variants that are no longer in window, keep closest one to cur_pos
        while len(var_queue) > 0 and var_queue[0].POS <= cur_pos - window:
            last_variant = var_queue.popleft()
            # test for end of interval
            if interval_start > 0 and len(var_queue) < density_threshold:
                intervals.append([record.CHROM, interval_start, last_variant.POS + window_len])
                interval_start = -1
        
        # test for start of interval
        var_queue.append(record)
        if interval_start < 0 and len(var_queue) >= density_threshold:
            # bed intervals indexed from 0
            # add 1 for interval to cover cur_pos
            interval_start = max(0, cur_pos - window_len + 1)

    if interval_start > 0:  # add last interval
        intervals.append([var_queue[-1].CHROM, interval_start, var_queue[-1].POS + window_len])
    return pd.DataFrame(intervals, columns=['seqname', 'start', 'end'])


if __name__ == '__main__':
    # test_filename = 'test/data/grch38_excerpt.fa'
    # self.test_filename = 'data/ref_genome/chr22.fa'
    # self.test_filename = 'data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa'
    # df = map_contigs(test_filename, buffer=65536)
    # df.to_csv('contig-coords-fai.csv')
    df = map_variant_density('./data/vcf/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz',
                    1000, 10)
    df.to_csv('chr22-1000-seq-10-variants.bed')

