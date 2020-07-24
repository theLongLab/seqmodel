import sys
sys.path.append('./src')
import unittest
import torch.nn as nn
import torch.nn.functional as F
import numpy.testing as npt

from seqmodel.task.loss import *


class Test_Loss(unittest.TestCase):

    def setUp(self):
        self.shape = (10, 4, 27)
        self.x = torch.rand(self.shape)
        self.target = torch.ones(self.shape)
        self.latent = torch.ones(self.shape)

    def test_NeighbourDistanceLoss(self):
        alternating = torch.tensor([[[1., - 1.] * 27] * 4] * 10)

        loss_fn = NeighbourDistanceLoss(distance_measure=nn.MSELoss())
        self.assertEqual(loss_fn(self.x, self.target, self.latent).item(), 0.)
        self.assertGreater(loss_fn(self.x, self.target, alternating).item(), 0.)

        loss_fn = NeighbourDistanceLoss(distance_measure=CosineSimilarityLoss())
        self.assertEqual(loss_fn(self.x, self.target, self.latent).item(), 0.)
        self.assertGreater(loss_fn(self.x, self.target, alternating).item(), 0.)

    def test_WeightedLoss(self):
        loss_fn = WeightedLoss({
            LossWrapper(nn.MSELoss()): 1.,
            LossWrapper(nn.BCELoss()): 1.,
        })

        self.assertEqual(loss_fn(self.x, self.target, self.latent),
            F.mse_loss(self.x, self.target) + F.binary_cross_entropy(self.x, self.target))
        loss_fn = WeightedLoss({
            LossWrapper(nn.MSELoss()): 0.5,
            LossWrapper(nn.BCELoss()): 2.,
        })
        loss = 0.5 * F.mse_loss(self.x, self.target) + \
                2. * F.binary_cross_entropy(self.x, self.target)
        self.assertEqual(loss_fn(self.x, self.target, self.latent).item(), loss.item())

    def test_LossWrapper(self):
        for fn in [nn.MSELoss(), nn.BCELoss()]:
            loss_fn = LossWrapper(fn)
            self.assertEqual(loss_fn(self.x, self.target, self.latent).item(),
                        fn(self.x, self.target).item())


if __name__ == '__main__':
    unittest.main()