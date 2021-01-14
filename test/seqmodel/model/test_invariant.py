import sys
sys.path.append('./src')
import unittest
import torch.nn as nn
import numpy.testing as npt

from seqmodel.model.invariant import *
from seqmodel.functional import complement, reverse, reverse_complement, one_hot, swap
from seqmodel.seqdata.mapseq import create_test_batch


class Test_Invariant(unittest.TestCase):

    def setUp(self):
        self.batch_size, self.seq_len = 5, 37
        self.x = one_hot(create_test_batch(self.batch_size, self.seq_len))

    def test_RCIConv1d(self):
        npt.assert_
        in_channels, out_channels, kernel_size = 4, 8, 7
        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=False, do_complement=False, padding=kernel_size // 2)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        conv.weight = rci_conv.weight
        conv.bias = rci_conv.bias
        self.assertEqual(rci_conv(self.x).shape, (self.batch_size, out_channels, self.seq_len))
        npt.assert_array_equal(rci_conv(self.x).detach(), conv(self.x).detach())

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=True, do_complement=False)
        y1 = rci_conv(reverse(self.x)).detach()
        y2 = reverse(rci_conv(self.x)).detach()
        # have to compare different halves of the array because the
        # normal and reversed output channels have switched places
        # assume output channels are grouped as (normal, reverse_complement, reverse, complement)
        # need to compare to (reverse, complement, normal, reverse_complement)
        npt.assert_array_equal(y1[:, :4, :], y2[:, 4:8, :])
        npt.assert_array_equal(y1[:, 4:8, :], y2[:, :4, :])

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=False, do_complement=True)
        y1 = complement(rci_conv(complement(self.x))).detach()
        y2 = rci_conv(self.x).detach()
        # don't need to switch because complementation is done along the channel dimension
        # this means every individual channel's position is mirrored relative to its complement
        # i.e. channel 0 is the complement of channel -1, 1 is the complement of -2, etc.
        npt.assert_array_equal(y1, y2)

        rci_conv = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=True, do_complement=True)
        self.check_equivariant(rci_conv)

        conv1 = RCIConv1d(in_channels, out_channels, kernel_size,
                do_reverse=True, do_complement=True, bias=False)
        conv2 = RCIConv1d(out_channels, out_channels, kernel_size, in_from_rci=True,
                do_reverse=True, do_complement=True, bias=False)
        conv3 = RCIConv1d(out_channels, out_channels + 4, kernel_size - 3, in_from_rci=True,
                do_reverse=True, do_complement=True, bias=False)
        layer = nn.Sequential(conv1, conv2, conv3)
        self.check_equivariant(layer)

    def check_equivariant(self, layer):
        y = layer(self.x).detach()
        y_c = complement(layer(complement(self.x)))
        y_r = swap(reverse(layer(reverse(self.x))))
        y_rc = swap(reverse_complement(layer(reverse_complement(self.x))))
        npt.assert_allclose(y_c.detach(), y)
        npt.assert_allclose(y_r.detach(), y)
        npt.assert_allclose(y_rc.detach(), y)

    def test_GroupConv1d(self):
        in_channels, out_channels, kernel_size = 4, 8, 7
        # conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        # gconv = GroupConv1d(in_channels, out_channels, kernel_size)
        # self.assertEqual(gconv.weight.shape, conv.weight.shape)
        # self.assertEqual(gconv.bias.shape, conv.bias.shape)
        # conv.weight = gconv.weight
        # conv.bias = gconv.bias
        # self.assertEqual(gconv(self.x).shape, conv(self.x).shape)
        # npt.assert_allclose(gconv(self.x).detach(), conv(self.x).detach(), rtol=1e-4)

    def test_v2GroupConv1d(self):
        in_channels, out_channels, kernel_size = 4, 8, 7
        conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        gconv = v2GroupConv1d(in_channels, out_channels, kernel_size)
        # self.assertEqual(gconv.weight.shape, conv.weight.shape)
        # self.assertEqual(gconv.bias.shape, conv.bias.shape)
        # conv.weight = gconv.weight
        # conv.bias = gconv.bias
        # self.assertEqual(gconv(self.x).shape, conv(self.x).shape)
        # npt.assert_allclose(gconv(self.x).detach(), conv(self.x).detach(), rtol=1e-4)
        # self.check_equivariant(gconv)


if __name__ == '__main__':
    unittest.main()
