import sys
sys.path.append('./src')
import unittest
import torch
import torch.nn as nn
import numpy.testing as npt

from seqmodel.model.conv import *


class Test_Conv(unittest.TestCase):

    def setUp(self):
        self.in_channels, self.out_channels, self.length, self.batches = 4, 10, 307, 7
        self.x = torch.randn(self.batches, self.in_channels)
        self.x = self.x.repeat(self.length, 1, 1).permute(1, 2, 0)

    def test_CellModule(self):
        conv_list = [
            nn.Conv1d(4, 4, 3),
            nn.Conv1d(4, 4, 3, padding=5),
            nn.Conv1d(4, 4, 5, stride=3),
            nn.Conv1d(4, 4, 4, dilation=2),
            nn.Conv1d(4, 10, 4, padding=3, stride=2, dilation=2),
        ]
        for conv in conv_list:
            layer = CellModule([conv])
            tensor = conv(self.x)
            self.assertEqual(layer.out_channels, tensor.shape[1])
            self.assertEqual(layer.out_seq_len(self.length), tensor.shape[2])
            npt.assert_array_equal(layer(self.x).detach(), tensor.detach())
        layer = CellModule(conv_list)
        tensor = self.x
        for conv in conv_list:
            tensor = conv(tensor)
        self.assertEqual(layer.out_channels, tensor.shape[1])
        self.assertEqual(layer.out_seq_len(self.length), tensor.shape[2])
        npt.assert_array_equal(layer(self.x).detach(), tensor.detach())

    def test_SeqFeedForward(self):
        layer = SeqFeedForward(self.in_channels, self.out_channels)
        out = layer(self.x).detach()
        self.assertEqual(out.shape, (self.batches, self.out_channels, self.length))
        for i in range(self.length):
            npt.assert_array_equal(out[:, :, i], out[:, :, 0])

        layer = SeqFeedForward(self.in_channels, self.out_channels, hidden_layers=2,
                            hidden_sizes=[5, 15], bias=False, activation_fn=nn.ReLU,
                            activation_on_last=True)
        out = layer(self.x).detach()
        self.assertEqual(out.shape, (self.batches, self.out_channels, self.length))
        npt.assert_array_equal((out >= 0.), torch.ones(out.shape, dtype=torch.bool))
        npt.assert_array_equal(layer(torch.zeros(self.x.shape)).detach(),
                                torch.zeros(out.shape).detach())

    def test_DilateConvCell(self):
        kernel_size = 3
        layer = DilateConvCell(self.in_channels, self.out_channels, kernel_size, dilation=3)
        out = layer(self.x).detach()
        self.assertEqual(out.shape, (self.batches, self.out_channels, self.length))
        #TODO: finish

    def test_DilateConvEncoder(self):
        kernel_size = 3
        layer = DilateConvEncoder(self.in_channels, kernel_size, dilation=3,
                                    n_cells=2, channel_growth_rate=2.)
        out = layer(self.x).detach()
        self.assertEqual(out.shape, (self.batches, self.in_channels * 2 * 2, self.length))
        #TODO: finish


if __name__ == '__main__':
    unittest.main()
