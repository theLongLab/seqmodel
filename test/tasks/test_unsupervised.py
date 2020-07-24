import sys
sys.path.append('./src')
import unittest
import numpy.testing as npt

from seqmodel.task.unsupervised import *
from seqmodel.seq.transform import one_hot, one_hot_to_index
from seqmodel.task.loss import LambdaLoss
from seqmodel.seq.mapseq import RandomRepeatSequence


class Test_Unsupervised(unittest.TestCase):

    def setUp(self):
        self.seq_len, self.batches = 39, 17
        self.data = RandomRepeatSequence(self.seq_len, self.batches, 2, repeat_len=2)
        self.x = torch.stack([self.data[i] for i in range(self.batches)], dim=0)
        self.identity = lambda x: x

    def test_GenericTask(self):
        task = GenericTask(self.identity, self.identity,
                    LambdaLoss(nn.MSELoss()), preprocess=self.identity)
        x = one_hot(self.x)
        latent, predicted = task.forward(x)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x)
        self.assertEqual(task.loss(x).item(), 0.)

        # need very large scores for cross entropy to softmax to 1.0 and 0.0
        large_num = 1000000.
        task = GenericTask(self.identity, lambda x: x * large_num,
                    LambdaLoss(nn.CrossEntropyLoss()))
        latent, predicted = task.forward(self.x)
        npt.assert_array_equal(latent, x)
        npt.assert_array_equal(predicted, x * large_num)
        self.assertEqual(task.loss(self.x).item(), 0.)
        #TODO test evaluate()

    def test_PredictMaskedTask(self):
        task = PredictMaskedTask(self.identity, self.identity, LambdaLoss(nn.CrossEntropyLoss()))
        generic_task = GenericTask(self.identity, self.identity, LambdaLoss(nn.CrossEntropyLoss()))
        npt.assert_array_equal(task.forward(self.x)[0].detach().numpy(),
                        generic_task.forward(self.x)[0].detach().numpy())
        npt.assert_array_equal(task.forward(self.x)[1].detach().numpy(),
                        generic_task.forward(self.x)[1].detach().numpy())
        # loss is nan because no positions are included in loss
        npt.assert_array_equal(task.loss(self.x).item(), float('nan'))

        mask_value = 3.
        task = PredictMaskedTask(self.identity, self.identity, LambdaLoss(nn.CrossEntropyLoss()),
                            mask_prop=1., mask_value=mask_value)
        npt.assert_array_equal(task.forward(self.x)[0],
                            torch.ones(one_hot(self.x).shape) * mask_value)
        # task = PredictMaskedTask(self.identity, self.identity, LambdaLoss(nn.MSELoss()),
        #                     random_prop=1.)
        # npt.assert_array_equal(task.forward(self.x)[0],
        #                     torch.ones(one_hot(self.x).shape) * mask_value)
        # task = PredictMaskedTask(self.identity, self.identity, LambdaLoss(nn.MSELoss()),
        #                     keep_prop=1.)
        # npt.assert_array_equal(task.forward(self.x)[0],
        #                     torch.ones(one_hot(self.x).shape) * mask_value)


if __name__ == '__main__':
    unittest.main()
