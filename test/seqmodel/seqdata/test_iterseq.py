import sys
sys.path.append('./src')
import unittest
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import numpy.testing as npt

from seqmodel.functional.transform import bioseq_to_index
from seqmodel.seqdata.dataset.datasets import FastaSequence, SeqIntervals
from seqmodel.seqdata.iterseq import *


class Test_StridedSeq(unittest.TestCase):

    def setUp(self):
        self.seq = FastaSequence('test/data/short.fa')
        self.seqs = [bioseq_to_index(x[:len(x)]) for x in self.seq.fasta.values()]

    def test_StridedSequence(self):
        dataset = StridedSequence(self.seq, 3, sequential=True)
        data = [x for x in dataset]
        self.assertEqual(len(data), 50 - 2 - 2)
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[1][-3:])

        dataset = StridedSequence(self.seq, 3, stride=3, start_offset=1)
        data = [x for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][1:4])
        npt.assert_array_equal(data[1], self.seqs[0][4:7])
        npt.assert_array_equal(data[-1], self.seqs[1][-4:-1])

    def test_StridedSequence_intervals(self):
        intervals = SeqIntervals.from_cols(['seq1____'], [0], [30])
        dataset = StridedSequence(self.seq, 3, sequential=True, include_intervals=intervals)
        data = [x for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])

        intervals = SeqIntervals.from_cols(
            ['seq2___', 'seq2___', 'seq1____', 'seq1____', 'seq1____'],
            [5, 17, 2, 15, 20],
            [9, 18, 5, 16, 30])
        dataset = StridedSequence(self.seq, 3, sequential=True, include_intervals=intervals)
        data = [x for x in dataset]
        self.assertEqual(len(data), 11)
        npt.assert_array_equal(data[0], self.seqs[1][5:8])
        npt.assert_array_equal(data[1], self.seqs[1][6:9])
        npt.assert_array_equal(data[2], self.seqs[0][2:5])
        npt.assert_array_equal(data[3], self.seqs[0][20:23])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])

        intervals = SeqIntervals.from_bed_file('data/ref_genome/grch38_contig_intervals.bed')
        dataset = StridedSequence.from_file('data/ref_genome/p12/assembled_chr/GRCh38_p12_assembled_chr.fa',
                                100, sequential=True, include_intervals=intervals)
        for i in dataset:
            self.assertFalse(np.any((i == 4).numpy()))
            break

if __name__ == '__main__':
    unittest.main()
