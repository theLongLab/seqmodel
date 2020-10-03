import sys
sys.path.append('./src')
import unittest
import random
from Bio import SeqIO
from Bio.Seq import Seq
import torch
import torch.nn as nn
import numpy.testing as npt

from seqmodel.functional.transform import *
from seqmodel.seqdata.mapseq import random_bioseq


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
        predictions = softmax_to_index(one_hot(self.indexes).float(), threshold_score= -2.)
        self.assertEqual(predictions.shape, (self.batches, self.seq_len))
        npt.assert_array_equal(predictions, self.indexes)
        predictions = softmax_to_index(one_hot(self.indexes).float(), threshold_score=2.)
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

    def test_swap(self):
        y1 = self.test_1h_tensor
        y2 = swap(y1)
        npt.assert_array_equal(y1[:, :2, :], y2[:, 2:4, :])
        npt.assert_array_equal(y1[:, 2:4, :], y2[:, :2, :])

    def test_Compose(self):
        fn = Compose(one_hot, one_hot_to_index)
        npt.assert_array_equal(fn(self.test_indexes), self.test_indexes)
        fn = Compose(index_to_bioseq, bioseq_to_index)
        npt.assert_array_equal(fn(self.indexes.flatten()), self.indexes.flatten())
        fn = Compose(one_hot, reverse, complement, reverse_complement)
        npt.assert_array_equal(fn(self.indexes), one_hot(self.indexes))

    def test_flank(self):
        x = torch.randn(10)
        y = flank(x)
        npt.assert_array_equal(y, x)
        start = torch.zeros(7)
        y = flank(x, start)
        npt.assert_array_equal(y[:7], start)
        npt.assert_array_equal(y[7:], x)
        end = torch.ones(7)
        y = flank(x, end_flank=end)
        npt.assert_array_equal(y[:-7], x)
        npt.assert_array_equal(y[-7:], end)
        y = flank(x, start, end)
        npt.assert_array_equal(y[:7], start)
        npt.assert_array_equal(y[7:-7], x)
        npt.assert_array_equal(y[-7:], end)

    def test_single_split(self):
        a, b = single_split(self.indexes)
        npt.assert_array_equal(a, self.indexes[:, :(self.seq_len // 2)])
        npt.assert_array_equal(b, self.indexes[:, (self.seq_len // 2):])
        a, b = single_split(self.indexes, prop=0.1)
        npt.assert_array_equal(a, self.indexes[:, :(self.seq_len // 10)])
        npt.assert_array_equal(b, self.indexes[:, (self.seq_len // 10):])

    def test_permute(self):
        def rows_equal(tensor_1, tensor_2, mask=None):
            if mask is None:
                mask = torch.ones(tensor_1.shape[0], dtype=torch.bool)
            is_equal = []
            for i, do_check in enumerate(mask):
                if do_check:
                    is_equal.append(torch.all(tensor_1[i,:] == tensor_2[i,:]))
            return torch.tensor(is_equal)

        is_permuted, x = permute(self.indexes, prop=0)
        npt.assert_array_equal(is_permuted, torch.zeros(self.batches, dtype=torch.bool))
        self.assertTrue(torch.all(rows_equal(x, self.indexes)))
        npt.assert_array_equal(x, self.indexes)  # this does the same thing, but tests rows_equal

        is_permuted, x = permute(self.indexes, prop=1.)
        npt.assert_array_equal(is_permuted, torch.ones(self.batches, dtype=torch.bool))
        self.assertFalse(torch.any(rows_equal(x, self.indexes)))

        is_permuted, x = permute(self.indexes)
        self.assertTrue(torch.sum(is_permuted), self.batches // 2)
        is_not_permuted = torch.logical_not(is_permuted)
        self.assertTrue(torch.all(rows_equal(x, self.indexes, mask=is_not_permuted)))
        same = torch.masked_select(x.permute(1, 0), is_not_permuted)
        ref = torch.masked_select(self.indexes.permute(1, 0), is_not_permuted)
        npt.assert_array_equal(same, ref)  # double check the above line
        self.assertFalse(torch.any(rows_equal(x, self.indexes, mask=is_permuted)))


if __name__ == '__main__':
    unittest.main()
