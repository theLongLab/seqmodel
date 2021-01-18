import sys
sys.path.append('./src')
import unittest
import torch
import numpy.testing as npt

from seqmodel.seqdata.contig import *


class Test_IO(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test/data/short.fa'

    def test_map_contigs(self):
        df = map_contigs(self.test_filename, buffer=65536)
        #TODO unfinished

    def test_map_variant_density(self):
        table = map_variant_density('./test/data/sample.vcf', 1, 1)
        npt.assert_array_equal(table['seqname'], ['chr1']*7 + ['chr2'])
        sample_variant_pos = torch.tensor([10003, 10005, 10007, 10009, 10011, 10020, 20000, 10000])
        npt.assert_array_equal(table['start'], sample_variant_pos)
        npt.assert_array_equal(table['end'], sample_variant_pos + 1)
        table = map_variant_density('./test/data/sample.vcf', 2, 1)
        npt.assert_array_equal(table['seqname'], ['chr1']*3 + ['chr2'])
        npt.assert_array_equal(table['start'], [10002, 10019, 19999, 9999])
        npt.assert_array_equal(table['end'], [10013, 10022, 20002, 10002])
        table = map_variant_density('./test/data/sample.vcf', 5, 1, min_gap_between_intervals=0)
        npt.assert_array_equal(table['seqname'], ['chr1']*3 + ['chr2'])
        npt.assert_array_equal(table['start'], [9999, 10016, 19996, 9996])
        npt.assert_array_equal(table['end'], [10016, 10025, 20005, 10005])
        table = map_variant_density('./test/data/sample.vcf', 5, 1, min_gap_between_intervals=5)
        npt.assert_array_equal(table['seqname'], ['chr1']*2 + ['chr2'])
        npt.assert_array_equal(table['start'], [9999, 19996, 9996])
        npt.assert_array_equal(table['end'], [10025, 20005, 10005])


if __name__ == '__main__':
    unittest.main()