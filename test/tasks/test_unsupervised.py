import sys
sys.path.append('./src')
import unittest
import numpy.testing as npt

from seqmodel.task.unsupervised import *
from seqmodel.seq.transform import one_hot, one_hot_to_index
from seqmodel.task.loss import LambdaLoss
from seqmodel.seq.mapseq import RandomRepeatSequence, create_test_batch


class Test_Unsupervised(unittest.TestCase):

    def setUp(self):
        self.x = create_test_batch(39, 17)
        self.identity = lambda x: x

    def test_GenericTask(self):
        task = GenericTask(self.identity, self.identity,
                    LambdaLoss(nn.MSELoss()), preprocess=self.identity)
        x = one_hot(self.x)
        predicted, latent = task(x)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x)
        self.assertEqual(task.loss(x).item(), 0.)

        # need very large scores for cross entropy to softmax to 1.0 and 0.0
        large_num = 1000000.
        task = GenericTask(self.identity, lambda x: x * large_num,
                    LambdaLoss(nn.CrossEntropyLoss()))
        predicted, latent = task(self.x)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x * large_num)
        self.assertEqual(task.loss(self.x).item(), 0.)
        #TODO test evaluate()

    def test_PredictMaskedTask(self):
        null_task = PredictMaskedTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), allow_no_loss=True)
        generic_task = GenericTask(self.identity, self.identity, LambdaLoss(nn.CrossEntropyLoss()))
        npt.assert_array_equal(null_task(self.x)[0].detach().numpy(),
                        generic_task(self.x)[0].detach().numpy())
        npt.assert_array_equal(null_task(self.x)[1].detach().numpy(),
                        generic_task(self.x)[1].detach().numpy())
        # loss is nan because no positions are included in loss
        npt.assert_array_equal(null_task.loss(self.x).item(), float('nan'))
        # at least one position is inserted to avoid nan loss
        task = PredictMaskedTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), allow_no_loss=False)
        self.assertGreater(task.loss(self.x).item(), 0.)

        mask_value = 3.
        task = PredictMaskedTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), mask_prop=1., mask_value=mask_value)
        npt.assert_array_equal(task(self.x)[0], torch.ones(one_hot(self.x).shape)*mask_value)
        npt.assert_array_equal(task.loss(self.x).item(),
                    nn.CrossEntropyLoss()(torch.ones(one_hot(self.x).shape)*mask_value, self.x))

        task = PredictMaskedTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), random_prop=1.)
        self.assertGreater(task.loss(self.x).item(), 0.)

        task = PredictMaskedTask(self.identity, self.identity,
                    LambdaLoss(nn.CrossEntropyLoss()), keep_prop=1.)
        npt.assert_array_equal(task(self.x)[0], null_task(self.x)[0])
        npt.assert_array_equal(task(self.x)[1], null_task(self.x)[1])
        npt.assert_array_equal(task.loss(self.x).item(), generic_task.loss(self.x).item())


if __name__ == '__main__':
    unittest.main()
