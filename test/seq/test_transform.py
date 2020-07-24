import sys
sys.path.append('./src')
import unittest
import random
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import torch.nn as nn
import numpy.testing as npt

from seqmodel.seq.transform import *
from seqmodel.seq.mapseq import random_bioseq


class Test_Transforms(unittest.TestCase):

    def setUp(self):
        self.test_indexes = torch.tensor([i for i in range(len(INDEX_TO_BASE))])
        self.test_seq = Seq(''.join(INDEX_TO_BASE))
        self.test_1h_tensor = torch.tensor([[[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1],]])
        self.batches, self.seq_len = 5, 374
        self.bioseq = random_bioseq(self.batches * self.seq_len)
        self.indexes = bioseq_to_index(self.bioseq).reshape(self.batches, self.seq_len)

    def test_bioseq_to_index(self):
        self.assertEqual(index_to_bioseq(self.test_indexes), self.test_seq)
        npt.assert_array_equal(bioseq_to_index(self.test_seq), self.test_indexes)
        self.assertEqual(index_to_bioseq(bioseq_to_index(self.bioseq)), self.bioseq)
        npt.assert_array_equal(bioseq_to_index(index_to_bioseq(self.indexes.flatten())),
                                self.indexes.flatten())

    def test_one_hot(self):
        tensor = one_hot(self.test_indexes.reshape(1, len(self.test_indexes)))
        npt.assert_array_equal(tensor, self.test_1h_tensor)
        tensor = one_hot(self.indexes)
        self.assertEqual(tensor.shape, (self.batches, N_BASE, self.seq_len))
        npt.assert_array_equal(one_hot_to_index(tensor), self.indexes)

    def test_softmax_to_index(self):
        tensor = self.test_1h_tensor.float()
        predictions = softmax_to_index(tensor)
        self.assertEqual(predictions.shape, (1, len(self.test_indexes)))
        npt.assert_array_equal(predictions, one_hot_to_index(tensor))
        predictions = softmax_to_index(one_hot(self.indexes).float(), prediction_threshold= -2.)
        self.assertEqual(predictions.shape, (self.batches, self.seq_len))
        npt.assert_array_equal(predictions, self.indexes)
        predictions = softmax_to_index(one_hot(self.indexes).float(), prediction_threshold=2.)
        npt.assert_array_equal(predictions, np.ones(predictions.shape) * EMPTY_INDEX)

    def test_reverse_complement(self):
        indexes = one_hot_to_index(complement(self.test_1h_tensor))
        npt.assert_array_equal(index_to_bioseq(indexes.flatten()), self.test_seq.complement())
        indexes = one_hot_to_index(reverse_complement(self.test_1h_tensor))
        npt.assert_array_equal(index_to_bioseq(indexes.flatten()), self.test_seq.reverse_complement())
        for tensor in [self.test_1h_tensor, one_hot(self.indexes)]:
            npt.assert_array_equal(complement(reverse(complement(tensor))), reverse(tensor))
            npt.assert_array_equal(reverse(complement(reverse(tensor))), complement(tensor))
            npt.assert_array_equal(reverse(complement(tensor)), reverse_complement(tensor))
            npt.assert_array_equal(reverse_complement(complement(reverse((tensor)))), tensor)

    def test_LambdaModule(self):
        fn = LambdaModule(one_hot, one_hot_to_index)
        npt.assert_array_equal(fn(self.test_indexes), self.test_indexes)
        fn = LambdaModule(index_to_bioseq, bioseq_to_index)
        npt.assert_array_equal(fn(self.indexes.flatten()), self.indexes.flatten())
        fn = LambdaModule(one_hot, reverse, complement, reverse_complement)
        npt.assert_array_equal(fn(self.indexes), one_hot(self.indexes))


if __name__ == '__main__':
    unittest.main()
