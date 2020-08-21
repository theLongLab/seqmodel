import sys
sys.path.append('./src')
import unittest
from pyfaidx import Fasta

from seqmodel.seq.iterseq import *


class Test_iterseq(unittest.TestCase):

    def setUp(self):
        self.fasta = Fasta('test/data/short.fa')

    def test_IterSequence(self):
        dataset = IterSequence(self.fasta, 3, 2)
        print([x for x in dataset])
        pass  #TODO unfinished


if __name__ == '__main__':
    unittest.main()