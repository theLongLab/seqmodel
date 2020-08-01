import sys
sys.path.append('./src')
import unittest
import torch.nn as nn
import torch.nn.functional as F
import numpy.testing as npt

from seqmodel.task.task import *
from seqmodel.seq.transform import one_hot
from seqmodel.seq.mapseq import create_test_batch


class Test_Task(unittest.TestCase):

    def setUp(self):
        self.shape = (10, 4, 27)
        self.x = torch.rand(self.shape)
        self.target = torch.ones(self.shape)
        self.latent = torch.ones(self.shape)
        self.batch = create_test_batch(39, 17)
        self.identity = lambda x: x

    def test_GenericTask(self):
        task = GenericTask(self.identity, self.identity,
                    LambdaLoss(nn.MSELoss()), preprocess=self.identity)
        x = one_hot(self.batch)
        predicted, latent = task(x)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x)
        self.assertEqual(task.loss(x)[3].item(), 0.)

        # need very large scores for cross entropy to softmax to 1.0 and 0.0
        large_num = 1e6
        task = GenericTask(self.identity, lambda x: x * large_num,
                    LambdaLoss(nn.CrossEntropyLoss()))
        predicted, latent = task(self.batch)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x * large_num)
        self.assertEqual(task.loss(self.batch)[3].item(), 0.)
        #TODO test evaluate()

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
            LambdaLoss(nn.MSELoss()): 1.,
            LambdaLoss(nn.BCELoss()): 1.,
        })

        self.assertEqual(loss_fn(self.x, self.target, self.latent),
            F.mse_loss(self.x, self.target) + F.binary_cross_entropy(self.x, self.target))
        loss_fn = WeightedLoss({
            LambdaLoss(nn.MSELoss()): 0.5,
            LambdaLoss(nn.BCELoss()): 2.,
        })
        loss = 0.5 * F.mse_loss(self.x, self.target) + \
                2. * F.binary_cross_entropy(self.x, self.target)
        self.assertEqual(loss_fn(self.x, self.target, self.latent).item(), loss.item())

    def test_LambdaLoss(self):
        for fn in [nn.MSELoss(), nn.BCELoss()]:
            loss_fn = LambdaLoss(fn)
            self.assertEqual(loss_fn(self.x, self.target, self.latent).item(),
                        fn(self.x, self.target).item())


if __name__ == '__main__':
    unittest.main()
