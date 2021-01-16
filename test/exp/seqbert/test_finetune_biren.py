import sys
sys.path.append('./src')
import unittest
import torch
import torch.nn as nn
import numpy.testing as npt

from exp.seqbert.finetune_biren import *
from seqmodel.seqdata.iterseq import StridedSequence
from seqmodel.functional import bioseq_to_index


class Test_LabelRandomizer(unittest.TestCase):

    def setUp(self):
        self.seqfile = FastaFile('test/data/enhancers.fa')
        self.dataset = StridedSequence(self.seqfile, 60,
                        sequential=True, transform=bioseq_to_index)
        self.labels = [x[1] for x in self.dataset]

    def tearDown(self):
        self.seqfile.fasta.close()

    def test_BatchProcessor_split(self):
        randomizer = LabelRadomizer(0.)
        labels = [randomizer.transform(k, c)[0] for k, c in self.labels]
        randomizer = LabelRadomizer(1.)
        random1 = [randomizer.transform(k, c)[0] for k, c in self.labels]
        equal = torch.tensor(labels) == torch.tensor(random1)
        self.assertFalse(torch.all(equal))
        self.assertAlmostEqual(torch.sum(equal).item() / len(labels), 0.5, 1)
        
        randomizer = LabelRadomizer(1.)
        random2 = [randomizer.transform(k, c)[0] for k, c in self.labels]
        npt.assert_array_equal(random1, random2)
        randomizer = LabelRadomizer(0.5)
        random = [randomizer.transform(k, c)[0] for k, c in self.labels]
        equal = torch.tensor(labels) == torch.tensor(random)
        self.assertAlmostEqual(torch.sum(equal).item() / len(labels), 0.75, 1)

        dataset = StridedSequence(self.seqfile, 60,
                        sequential=True, label_transform=randomizer.transform)
        npt.assert_array_equal([x[1][0] for x in dataset], random)


if __name__ == '__main__':
    unittest.main()
