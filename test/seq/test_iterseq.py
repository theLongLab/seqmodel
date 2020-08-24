import sys
sys.path.append('./src')
import unittest
from pyfaidx import Fasta
import numpy.testing as npt

from seqmodel.seq.transform import bioseq_to_index
from seqmodel.seq.iterseq import *


class Test_iterseq(unittest.TestCase):

    def setUp(self):
        self.fasta_file = 'test/data/short.fa'
        self.seqs = [bioseq_to_index(x[:len(x)]) for x in fasta_from_file(self.fasta_file).values()]

    def test_IterSequence(self):
        dataset = IterSequence(self.fasta_file, 3, sequential=True)
        data = [x for x in dataset]
        self.assertEqual(len(data), 50 - 2 - 2)
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[1][-3:])

        dataset = IterSequence(self.fasta_file, 3, stride=3, start_offset=1)
        data = [x for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][1:4])
        npt.assert_array_equal(data[1], self.seqs[0][4:7])
        npt.assert_array_equal(data[-1], self.seqs[1][-4:-1])

        intervals = {
            'chr': ['seq1____'],
            'start': [0],
            'end': [30],
        }
        dataset = IterSequence(self.fasta_file, 3, sequential=True, included_intervals=intervals)
        data = [x for x in dataset]
        npt.assert_array_equal(data[0], self.seqs[0][:3])
        npt.assert_array_equal(data[1], self.seqs[0][1:4])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])

        intervals = {
            'chr': ['seq2___', 'seq2___', 'seq1____', 'seq1____', 'seq1____'],
            'start': [5, 17, 2, 15, 20],
            'end': [9, 18, 5, 16, 30],
        }
        dataset = IterSequence(self.fasta_file, 3, sequential=True, included_intervals=intervals)
        data = [x for x in dataset]
        self.assertEqual(len(data), 11)
        npt.assert_array_equal(data[0], self.seqs[1][5:8])
        npt.assert_array_equal(data[1], self.seqs[1][6:9])
        npt.assert_array_equal(data[2], self.seqs[0][2:5])
        npt.assert_array_equal(data[3], self.seqs[0][20:23])
        npt.assert_array_equal(data[-1], self.seqs[0][-3:])


if __name__ == '__main__':
    unittest.main()