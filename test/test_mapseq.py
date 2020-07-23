import sys
sys.path.append('./src')
import unittest
import random
from Bio import SeqIO
from Bio.Seq import Seq
import torch.nn as nn
import numpy.testing as npt

from seqmodel.seq.mapseq import *


class Test_Mapseq(unittest.TestCase):

    def test_random_seq(self):
        length = 1
        seq = random_bioseq(length)
        self.assertEqual(len(seq), length)

if __name__ == '__main__':
    unittest.main()
