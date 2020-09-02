import sys
sys.path.append('./src')
import unittest

from seqmodel.seqdata.dataset.datasets import *


class Test_DownloadableDataset(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test/data/region.txt'

    def test_cache_path(self):
        pass  #TODO


class Test_SeqIntervals(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test/data/region.txt'

    def test_union(self):
        pass  #TODO

    def test_intersect(self):
        pass  #TODO

    def test_remove(self):
        pass  #TODO


if __name__ == '__main__':
    unittest.main()
