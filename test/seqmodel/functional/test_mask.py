import sys
sys.path.append('./src')
import unittest
import torch
import numpy.testing as npt

from seqmodel.functional.mask import *
from seqmodel.functional.transform import one_hot, one_hot_to_index
from seqmodel.seqdata.mapseq import RandomRepeatSequence, create_test_batch


class Test_PositionMask(unittest.TestCase):

    def setUp(self):
        self.x = create_test_batch(39, 17)
        self.mask = PositionMask()
        self.identity = lambda x: x

    def test_generate(self):
        self.mask.generate(self.x, require_loss_pos=False)
        mask1 = self.mask.mask_val
        self.assertEqual(mask1.shape, self.x.shape)
        npt.assert_array_equal(mask1, torch.zeros(self.x.shape, dtype=torch.int8))
        self.mask.generate(self.x, require_loss_pos=True)
        self.assertEqual(self.mask.mask_val[0, 0].item(), PositionMask._KEEP_INDEX)
        self.mask = PositionMask(mask_prop=1.)
        self.mask.generate(self.x, flank_start=3, flank_end=2)
        mask1 = self.mask.mask_val
        npt.assert_array_equal(mask1[:, :3], torch.zeros_like(self.x)[:, :3])
        npt.assert_array_equal(mask1[:, 3:-2], torch.ones_like(self.x)[:, 3:-2])
        npt.assert_array_equal(mask1[:, -2:], torch.zeros_like(self.x)[:, -2:])

    def test_set_mask_props(self):
        def assert_equal_props(mask_1, mask_2):
            self.assertEqual(mask_1._mask_cutoff, mask_2._mask_cutoff)
            self.assertEqual(mask_1._random_cutoff, mask_2._random_cutoff)
            self.assertEqual(mask_1._keep_cutoff, mask_2._keep_cutoff)

        self.mask.set_mask_props(1., 0., 0.)
        self.assertEqual(self.mask._mask_cutoff, 0.)
        self.assertEqual(self.mask._random_cutoff, 1.)
        self.assertEqual(self.mask._keep_cutoff, 1.)
        assert_equal_props(self.mask, PositionMask(1., 0., 0.))
        self.mask.generate(self.x)
        npt.assert_array_equal(self.mask.mask_val, torch.ones(self.x.shape, dtype=torch.int8) * PositionMask._MASK_INDEX)

        self.mask.set_mask_props(0., 1., 0.)
        self.assertEqual(self.mask._mask_cutoff, 0.)
        self.assertEqual(self.mask._random_cutoff, 0.)
        self.assertEqual(self.mask._keep_cutoff, 1.)
        assert_equal_props(self.mask, PositionMask(0., 1., 0.))
        self.mask.generate(self.x)
        npt.assert_array_equal(self.mask.mask_val, torch.ones(self.x.shape, dtype=torch.int8) * PositionMask._RANDOM_INDEX)

        self.mask.set_mask_props(0., 0., 1.)
        self.assertEqual(self.mask._mask_cutoff, 0.)
        self.assertEqual(self.mask._random_cutoff, 0.)
        self.assertEqual(self.mask._keep_cutoff, 0.)
        assert_equal_props(self.mask, PositionMask(0., 0., 1.))
        self.mask.generate(self.x)
        npt.assert_array_equal(self.mask.mask_val, torch.ones(self.x.shape, dtype=torch.int8) * PositionMask._KEEP_INDEX)

        self.mask.set_mask_props(0.1, 0.2, 0.5)
        self.assertAlmostEqual(self.mask._mask_cutoff, 0.2)
        self.assertAlmostEqual(self.mask._random_cutoff, 0.3)
        self.assertAlmostEqual(self.mask._keep_cutoff, 0.5)
        assert_equal_props(self.mask, PositionMask(0.1, 0.2, 0.5))
        self.mask.generate(self.x)

        mask1 = self.mask.mask_val
        self.mask.generate(self.x)
        mask2 = self.mask.mask_val
        self.assertFalse(torch.all(mask1 == mask2))
        n_mask = torch.sum(self.mask.mask_val == PositionMask._MASK_INDEX)
        n_random = torch.sum(self.mask.mask_val == PositionMask._RANDOM_INDEX)
        n_keep = torch.sum(self.mask.mask_val == PositionMask._KEEP_INDEX)
        self.assertLess(n_mask, n_random)
        self.assertLess(n_random, n_keep)

    def test_randomize_input(self):
        self.mask.set_mask_props(random_prop=1.)
        self.mask.generate(self.x)
        self.assertFalse(torch.all(self.mask.randomize_input(self.x) == self.x))
        self.mask.set_mask_props(random_prop=0.)
        self.mask.generate(self.x)
        npt.assert_array_equal(self.mask.randomize_input(self.x), self.x)

    def test_mask_fill(self):
        self.mask.set_mask_props(mask_prop=1.)
        self.mask.generate(self.x)
        value = 2.5
        x = self.mask.mask_fill(torch.zeros_like(one_hot(self.x)), fill_value=value)
        npt.assert_array_equal(x, torch.ones_like(x) * value)
        self.mask.set_mask_props(mask_prop=0.)
        self.mask.generate(self.x, require_loss_pos=False)
        x = self.mask.mask_fill(torch.zeros_like(one_hot(self.x)), fill_value=value)
        npt.assert_array_equal(x, torch.zeros_like(x))

        self.mask.mask_val = torch.zeros_like(self.x)
        self.mask.mask_val[0, 0] = PositionMask._MASK_INDEX
        x = self.mask.mask_fill(one_hot(self.x), 0)
        npt.assert_array_equal(x[0, :, 0], [0, 0, 0, 0])
        npt.assert_array_equal(x[1:, :, 1:], one_hot(self.x)[1:, :, 1:])

    def test_get(self):
        self.mask.mask_val = torch.zeros_like(self.x)
        self.mask.mask_val[0, 0] = PositionMask._MASK_INDEX
        x = self.mask.get()
        self.assertEqual(x[0, 0], float('-inf'))
        npt.assert_array_equal(x[1:, 1:], torch.zeros_like(self.x)[1:, 1:])

    def test_attn_mask(self):
        self.mask.set_mask_props(0., 0., 0.)
        x, mask = self.mask.attn_mask(self.x)
        npt.assert_array_equal(x, self.x)
        npt.assert_array_equal(mask, torch.zeros_like(self.x))
        self.mask.set_mask_props(keep_prop=1.)
        x, mask = self.mask.attn_mask(self.x)
        npt.assert_array_equal(x, self.x)
        npt.assert_array_equal(mask, torch.zeros_like(self.x))
        self.mask.set_mask_props(mask_prop=1.)
        x, mask = self.mask.attn_mask(self.x, mask_fill=True)
        npt.assert_array_equal(x, torch.ones_like(self.x) * 4)
        npt.assert_array_equal(mask, float('-inf'))
        self.mask.set_mask_props(random_prop=1.)
        x, mask = self.mask.attn_mask(self.x)
        self.assertFalse(torch.all(self.mask.randomize_input(self.x) == self.x))
        npt.assert_array_equal(mask, torch.zeros_like(self.x))

    def test_select(self):
        self.mask.mask_val = torch.ones_like(self.x) * PositionMask._NO_LOSS_INDEX
        self.mask.mask_val[0, 0] = PositionMask._MASK_INDEX
        self.mask.mask_val[0, 1] = PositionMask._KEEP_INDEX
        self.mask.mask_val[0, 2] = PositionMask._RANDOM_INDEX
        self.mask.mask_val[0, 3] = PositionMask._NO_LOSS_INDEX
        a, b = self.mask.select(self.x, one_hot(self.x))
        npt.assert_array_equal(a.shape, [3])
        npt.assert_array_equal(a, self.x[0, :3])
        npt.assert_array_equal(b.shape, [3, 4])
        npt.assert_array_equal(b, one_hot(self.x)[0, :, :3].permute(1, 0))
        self.mask.mask_val = torch.ones_like(self.x) * PositionMask._KEEP_INDEX
        a, b = self.mask.select(self.x, one_hot(self.x))
        npt.assert_array_equal(one_hot(a), b)
        npt.assert_array_equal(one_hot_to_index(b), a)


if __name__ == '__main__':
    unittest.main()
