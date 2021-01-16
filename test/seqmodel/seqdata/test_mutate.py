import sys
sys.path.append('./src')
import unittest
import torch
import numpy as np
import numpy.testing as npt

from seqmodel import BASE_TO_INDEX
from seqmodel.functional import bioseq_to_index
from seqmodel.seqdata.mapseq import random_bioseq
from seqmodel.seqdata.mutate import *
from seqmodel.seqdata.mapseq import RandomRepeatSequence


class Test_MutationModel(unittest.TestCase):

    def setUp(self):
        self.seq_len = 1000
        self.seq = bioseq_to_index(random_bioseq(self.seq_len))

    def test_FixedRateSubstitutionModel(self):
        model = FixedRateSubstitutionModel(global_rate=1.)
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        self.assertEqual(chrom, 'N/A')
        npt.assert_array_equal(pos.shape, self.seq.shape)
        npt.assert_array_equal(alt.shape, self.seq.shape)
        self.assertFalse(torch.any(alt == self.seq))
        model = FixedRateSubstitutionModel(global_rate=0.)
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        npt.assert_array_equal(pos.shape, [0])
        npt.assert_array_equal(alt.shape, [0])
        model = FixedRateSubstitutionModel(global_rate=0.5)
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        self.assertAlmostEqual(alt.size(0) / self.seq_len, 0.5, 1)
        self.assertAlmostEqual(pos.size(0) / self.seq_len, 0.5, 1)
        self.assertFalse(torch.any(alt == self.seq[pos]))
        npt.assert_array_equal(ref, self.seq[pos])

        A = BASE_TO_INDEX['A']
        G = BASE_TO_INDEX['G']
        C = BASE_TO_INDEX['C']
        T = BASE_TO_INDEX['T']
        model = FixedRateSubstitutionModel(transition_rate=1.)
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        self.assertTrue(torch.all(alt[self.seq == A] == G))
        self.assertTrue(torch.all(alt[self.seq == G] == A))
        self.assertTrue(torch.all(alt[self.seq == C] == T))
        self.assertTrue(torch.all(alt[self.seq == T] == C))
        model = FixedRateSubstitutionModel(transversion_rate=1.)
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        self.assertFalse(torch.any(alt[self.seq == A] == G))
        self.assertFalse(torch.any(alt[self.seq == G] == A))
        self.assertFalse(torch.any(alt[self.seq == C] == T))
        self.assertFalse(torch.any(alt[self.seq == T] == C))

        model = FixedRateSubstitutionModel(global_rate=0.3,
                nucleotide_rates={
                    'A': {'C': 0.8},
                    'T': {'A': 0.5, 'C': 0.},
                    'C': {'A': 0.5, 'G': 0.3, 'T': 0.2},
                    })
        chrom, pos, ref, alt = model.gen_variants(self.seq, 'N/A', 0)
        self.assertAlmostEqual(torch.sum((alt == A).float()).item() / self.seq_len,
                                (0.5 + 0.5 + 0.1) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == G).float()).item() / self.seq_len,
                                (0.1 + 0.1 + 0.3) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == C).float()).item() / self.seq_len,
                                (0.8 + 0. + 0.1) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == T).float()).item() / self.seq_len,
                                (0.1 + 0.2 + 0.1) * 0.25, 1)
        self.assertFalse(torch.any(self.seq[pos[(alt == C)]] == T))


if __name__ == '__main__':
    unittest.main()
