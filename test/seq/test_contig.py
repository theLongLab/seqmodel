import sys
sys.path.append('./src')
import unittest

from seqmodel.seq.contig import *


class Test_IO(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test/data/short.fa'

    def test_map_contigs(self):
        df = map_contigs(self.test_filename, buffer=65536)

if __name__ == '__main__':
    unittest.main()