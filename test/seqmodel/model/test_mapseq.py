import sys
sys.path.append('./src')
import unittest
import random
from Bio import SeqIO
from Bio.Seq import Seq
import torch.nn as nn
import numpy.testing as npt

from seqmodel.seqdata.mapseq import *
from seqmodel.functional.transform import index_to_bioseq


class Test_Mapseq(unittest.TestCase):

    def setUp(self):
        self.length = 314
        self.seq = random_bioseq(self.length)
        self.test_filename = 'test/data/grch38_excerpt.fa'
        self.test_len = 999 * 80

    def test_random_seq(self):
        self.assertEqual(len(self.seq), self.length)

    def test_RandomRepeatSequence(self):
        seq_len = 10
        data = RandomRepeatSequence(seq_len, self.length, 3, repeat_len=1)
        self.assertEqual(len(data), self.length)
        batch = data[0]
        self.assertEqual(len(batch), seq_len)
        self.assertEqual(batch[0], batch[1], batch[2])
        data = RandomRepeatSequence(seq_len, self.length, 2, repeat_len=2)
        self.assertEqual(batch[0], batch[2])
        self.assertEqual(batch[1], batch[3])

    def test_MapSequence(self):
        subseq_len = 31
        data_1 = MapSequence(self.seq, subseq_len, stride=1)
        data_2 = MapSequence(self.seq, subseq_len, overlap=subseq_len - 1)
        self.assertEqual(len(data_1), self.length - subseq_len + 1)
        self.assertEqual(len(data_2), len(data_1))
        self.assertEqual(index_to_bioseq(data_1[0]), self.seq[:subseq_len])
        npt.assert_array_equal(data_2[0], data_1[0])
        self.assertEqual(index_to_bioseq(data_1[len(data_1) - 1]), self.seq[-1 * subseq_len:])
        npt.assert_array_equal(data_2[len(data_2) - 1], data_1[len(data_1) - 1])

        data_1 = MapSequence(self.seq, subseq_len, stride=subseq_len)
        data_2 = MapSequence(self.seq, subseq_len, overlap=0)
        self.assertEqual(len(data_1), int(self.length / subseq_len))
        self.assertEqual(len(data_2), len(data_1))
        npt.assert_array_equal(data_2[0], data_1[0])
        npt.assert_array_equal(data_2[len(data_2) - 1], data_1[len(data_1) - 1])

        fasta_data = MapSequence.from_file(self.test_filename, subseq_len, remove_gaps=False)
        self.assertEqual(len(fasta_data), self.test_len - subseq_len + 1)
        ungapped = MapSequence.from_file(self.test_filename, subseq_len, remove_gaps=True)
        self.assertLess(len(ungapped), len(fasta_data))


if __name__ == '__main__':
    unittest.main()
