import sys
sys.path.append('./src')
import unittest
import torch
import torch.nn as nn
import numpy.testing as npt

from seqmodel.functional.log import *
from seqmodel.seqdata.mapseq import create_test_batch
from seqmodel.functional.transform import one_hot, index_to_bioseq, INDEX_TO_BASE


class Test_Conv(unittest.TestCase):

    def setUp(self):
        self.batch = create_test_batch(39, 17)
        self.large_number = 1e6

    def test_correct(self):
        x = one_hot(self.batch)
        npt.assert_array_equal(correct(x, self.batch),
                    torch.ones(self.batch.shape, dtype=torch.bool))
        self.assertEqual(n_correct(x, self.batch), self.batch.nelement())
        self.assertEqual(accuracy(x, self.batch), 1.)

        npt.assert_array_equal(correct(x, self.batch, threshold_score=2.),
                    torch.zeros(self.batch.shape, dtype=torch.bool))
        self.assertEqual(n_correct(x, self.batch, threshold_score=2.), 0)
        self.assertEqual(accuracy(x, self.batch, threshold_score=2.), 0.)

        x = one_hot(self.batch + 1)
        npt.assert_array_equal(correct(x, self.batch, threshold_score=0.5),
                    torch.zeros(self.batch.shape, dtype=torch.bool))
        self.assertEqual(n_correct(x, self.batch, threshold_score=0.5), 0)
        self.assertEqual(accuracy(x, self.batch, threshold_score=0.5), 0.)

    def test_prediction_histograms(self):
        n_bins = 7
        x = one_hot(self.batch) * self.large_number
        hist = prediction_histograms(x, self.batch, n_bins=n_bins)
        self.assertEqual(hist.shape, (2, 4, n_bins))
        npt.assert_array_equal(hist[0, :, :], torch.zeros(4, n_bins))
        npt.assert_array_equal(hist[1, :, :-1], torch.zeros(4, n_bins - 1))
        self.assertEqual(torch.sum(hist[:,:, -1]).item(), self.batch.nelement())
        normalized = normalize_histogram(hist)
        npt.assert_array_less(normalized, 1 + 1e-5)
        npt.assert_array_less(0 - 1e-5, normalized)
        self.assertEqual(torch.sum(normalized[:,:, -1]).item(), 4)

        x = torch.zeros(x.shape)
        hist = prediction_histograms(x, self.batch, n_bins=n_bins)
        npt.assert_array_equal(hist[:, :, 1:], torch.zeros(2, 4, n_bins - 1))
        self.assertEqual(torch.sum(hist[:,:, 0]).item(), self.batch.nelement())
        normalized = normalize_histogram(hist)
        npt.assert_array_less(normalized, 1 + 1e-5)
        npt.assert_array_less(0 - 1e-5, normalized)
        self.assertEqual(torch.sum(normalized[:,:, 0]).item(), 4)

        x = one_hot(torch.ones(self.batch.shape))
        hist = prediction_histograms(x, self.batch, n_bins=n_bins)
        npt.assert_array_equal(accuracy_per_class(hist, threshold_prob=0.), [0., 1., 0., 0.])
        npt.assert_array_equal(accuracy_per_class(hist, threshold_prob=0.5), [0., 0., 0., 0.])

        hist = prediction_histograms(x[1], self.batch[1], n_bins=n_bins)
        npt.assert_array_equal(accuracy_per_class(hist, threshold_prob=0.), [0., 1., 0., 0.])
        npt.assert_array_equal(accuracy_per_class(hist, threshold_prob=0.5), [0., 0., 0., 0.])

    def test_roc_auc(self):
        shape = (4, 1000)
        x = torch.zeros(shape)
        y = torch.ones(shape)
        self.assertEqual(roc_auc(x, y), 0.)
        y = torch.zeros(shape)
        self.assertEqual(roc_auc(x, y), 0.)
        x = torch.ones(shape)
        y = torch.randint(2, shape)
        x = torch.rand(shape)
        self.assertAlmostEqual(roc_auc(x, y), 0.5, places=1)
        y = (x > 0.5).to(torch.int)
        self.assertEqual(roc_auc(x, y), 1.)
        y = 1 - y
        self.assertEqual(roc_auc(x, y), 0.)

    def test_excerpt(self):
        x = torch.arange(10)
        y = excerpt(x, max_sizes=[5])[0]
        self.assertEqual(list(y.shape), [5])
        npt.assert_array_equal(y, x[:5])
        y = excerpt(x, max_sizes=[12], random_pos=True)[0]
        self.assertEqual(list(y.shape), [10])
        npt.assert_array_equal(y, x)
        y = excerpt(x, max_sizes=[5], random_pos=True)[0]
        self.assertEqual(y.shape, x[:5].shape)

        y1, y2 = excerpt(torch.arange(10), torch.arange(10), max_sizes=[5])
        npt.assert_array_equal(y1, x[:5])
        npt.assert_array_equal(y1, y2)
        x = torch.arange(20).view(2, 10)
        y1, y2 = excerpt(x[0], x, max_sizes=[2, 5])
        npt.assert_array_equal(y1, x[0:1, :5])
        npt.assert_array_equal(y2, x[:, :5])
        x = torch.arange(24).view(2, 3, 4)
        y1, y2 = excerpt(x[0], x, max_sizes=[2, 2, 5])
        npt.assert_array_equal(y1, x[0:1, :2, :])
        npt.assert_array_equal(y2, x[:, :2, :])
        y1, y2 = excerpt(x[0], x, max_sizes=[15])
        npt.assert_array_equal(y1, x[0, 0, :])
        npt.assert_array_equal(y2, x[0, 0, :])

    def test_summarize(self):
        x = one_hot(torch.zeros(1, 7)).permute(1, 0, 2) * self.large_number
        self.assertEqual(summarize(x, max_len=10),
            'A|@@@@@@@|\nG|       |\nC|       |\nT|       |')
        self.assertEqual(summarize(x, max_len=6),
            'A|@@@|\nG|   |\nC|   |\nT|   |')
        x = one_hot(torch.arange(4).view(1, 4)).permute(1, 0, 2) * self.large_number
        self.assertEqual(summarize(x, max_len=10),
            'A|@   |\nG| @  |\nC|  @ |\nT|   @|')
        x = one_hot(torch.zeros(2, 7)).permute(1, 0, 2) * self.large_number
        self.assertEqual(summarize(x, max_len=10),
            'A|@@@|@@@|\nG|   |   |\nC|   |   |\nT|   |   |')
        x = one_hot(self.batch).permute(1, 0, 2)
        output = summarize(self.batch, x, max_len=10).split('\n')
        self.assertEqual(len(output), 5)
        for substr in output:
            self.assertEqual(len(substr), 10)

        x = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(summarize(x), ' |....|')
        x = torch.zeros(1, 4, dtype=torch.bool)
        self.assertEqual(summarize(x), ' |!!!!|')
        x = torch.randint(4, [1, 12])
        self.assertEqual(summarize(x), ' |' + index_to_bioseq(x.flatten()) + '|')

        x = torch.tensor([float('nan')]* 12).reshape(4, 1, 3)
        self.assertEqual(summarize(x), 'A|XXX|\nG|XXX|\nC|XXX|\nT|XXX|')
        x = torch.randn_like(one_hot(self.batch), dtype=torch.float)
        hist = normalize_histogram(prediction_histograms(x, self.batch, n_bins=5))
        output = summarize(hist, col_labels=INDEX_TO_BASE, normalize_fn=None).split('\n')
        self.assertEqual(len(output), 3)
        for substr in output:
            self.assertEqual(len(substr), 4 * (5 + 1) + 2)
        output = summarize(one_hot_to_index(x) == self.batch, self.batch, x.permute(1, 0, 2),
                            max_len=90)
        output = output.split('\n')
        self.assertEqual(len(output), 6)
        col_len = self.batch.shape[1] + 1
        n_cols = (90 - 2) // col_len
        for substr in output:
            self.assertEqual(len(substr), n_cols * col_len + 2)


if __name__ == '__main__':
    unittest.main()
