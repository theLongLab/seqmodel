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
        self.identity = lambda x: x

    def test_generate_mask(self):
        mask = generate_mask(self.x, [0.], require_loss_pos=False)
        self.assertEqual(mask.shape, self.x.shape)
        npt.assert_array_equal(mask, torch.zeros(self.x.shape, dtype=torch.int8))
        mask = generate_mask(self.x, [0.], require_loss_pos=True)
        self.assertEqual(mask[-1, -1].item(), 1)
        mask = generate_mask(self.x, [1.], require_loss_pos=True)
        npt.assert_array_equal(mask, torch.ones(self.x.shape, dtype=torch.int8))

    def test_set_mask_props(self):
        props = [1., 0., 0.]
        cutoffs = prop_cutoffs(props)
        npt.assert_array_equal(cutoffs, [1., 1., 1.])
        mask = generate_mask(self.x, props)
        npt.assert_array_equal(mask, torch.ones(self.x.shape, dtype=torch.int8))

        props = [0., 1., 0.]
        cutoffs = prop_cutoffs(props)
        npt.assert_array_equal(cutoffs, [0., 1., 1.])
        mask = generate_mask(self.x, props)
        npt.assert_array_equal(mask, torch.ones(self.x.shape, dtype=torch.int8) * 2)

        props = [0., 0., 1.]
        cutoffs = prop_cutoffs(props)
        npt.assert_array_equal(cutoffs, [0., 0., 1.])
        mask = generate_mask(self.x, props)
        npt.assert_array_equal(mask, torch.ones(self.x.shape, dtype=torch.int8) * 3)

        props = [0.1, 0.2, 0.5]
        cutoffs = prop_cutoffs(props)
        npt.assert_allclose(cutoffs, [0.1, 0.3, 0.8])
        mask = generate_mask(self.x, props)
        n_0 = torch.sum(mask == 0)  # ~= 0.2
        n_1 = torch.sum(mask == 1)  # ~= 0.1
        n_2 = torch.sum(mask == 2)  # ~= 0.2
        n_3 = torch.sum(mask == 3)  # ~= 0.5
        self.assertLess(n_1, n_0)
        self.assertLess(n_1, n_2)
        self.assertLess(n_0, n_3)
        self.assertLess(n_2, n_3)

    def test_mask_randomize(self):
        mask = generate_mask(self.x, [1.])
        self.assertFalse(torch.any(mask_randomize(self.x, mask == 1, 4) == self.x))
        npt.assert_array_equal(mask_randomize(self.x, mask != 1, 4), self.x)


    def test_mask_fill(self):
        mask = generate_mask(self.x, [1.])
        value = 2.5
        x = mask_fill(torch.zeros_like(one_hot(self.x)), mask == 1, fill_value=value)
        npt.assert_array_equal(x, torch.ones_like(x) * value)
        x = mask_fill(torch.zeros_like(one_hot(self.x)), mask != 1, fill_value=value)
        npt.assert_array_equal(x, torch.zeros_like(x))

        mask[0, 0] = 0
        x = mask_fill(one_hot(self.x), mask == 0, value)
        npt.assert_array_equal(x[0, :, 0], [value, value, value, value])
        npt.assert_array_equal(x[1:, :, 1:], one_hot(self.x)[1:, :, 1:])

    def test_select(self):
        mask = generate_mask(self.x, [0.])
        a, b = torch.randint(self.x.shape[0], [1]).item(), torch.randint(self.x.shape[1], [1]).item()
        mask[a:, b:] = 1
        x = mask_select(self.x, mask == 1)
        npt.assert_array_equal(x, self.x[a:, b:].flatten())
        x = mask_select(one_hot(self.x), mask == 1)
        npt.assert_array_equal(x, one_hot(self.x[a:, b:].flatten()))
        mask[:a, :b] = 2
        x = mask_select(self.x, mask == 2)
        npt.assert_array_equal(x, self.x[:a, :b].flatten())
        x = mask_select(one_hot(self.x), mask == 2)
        npt.assert_array_equal(x, one_hot(self.x[:a, :b].flatten()))


if __name__ == '__main__':
    unittest.main()
