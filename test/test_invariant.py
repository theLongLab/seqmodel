import sys
sys.path.append('./src')
import unittest

import torch.nn as nn
import numpy.testing as npt

from seqmodel.model.invariant import *


class Test_Invariant(unittest.TestCase):

    def setUp(self):
        pass

    def test_RCIConv1d(self):
        in_channels, out_channels, kernel_size = 4, 8, 3
        batch_size, seq_len = 5, 7

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=False, do_complement=False)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        conv.weight = rci_conv.weight
        conv.bias = rci_conv.bias

        x = torch.randn(batch_size, in_channels, seq_len)
        self.assertEqual(rci_conv(x).shape, conv(x).shape)
        npt.assert_array_equal(rci_conv(x).detach(), conv(x).detach())

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=True, do_complement=False)

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=False, do_complement=True)


if __name__ == '__main__':
    unittest.main()
