import sys
sys.path.append('./src')
import unittest
import vcf
import torch
import numpy as np
import numpy.testing as npt

from seqmodel import BASE_TO_INDEX
from seqmodel.seqdata.iterseq import fasta_from_file
from seqmodel.functional import bioseq_to_index
from seqmodel.seqdata.mapseq import random_bioseq
from seqmodel.seqdata.mapseq import RandomRepeatSequence
from seqmodel.seqdata.variant import *


class Test_Variant(unittest.TestCase):

    def setUp(self):
        self.seq_len = 1000
        self.seq = bioseq_to_index(random_bioseq(self.seq_len))
        self.chr = 'chr1'
        self.pos = 10000
        fasta = 'test/data/grch38_excerpt.fa'
        self.ref_seq = fasta_from_file(fasta)
        self.variants = vcf.Reader(filename='test/data/sample.vcf.gz')

    def tearDown(self):
        self.ref_seq.close()

    def test_FixedRateSubstitutionModel(self):
        model = FixedRateSubstitutionModel(global_rate=1.)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
        npt.assert_array_equal(chrom, 'N/A')
        npt.assert_array_equal(pos.shape, self.seq.shape)
        npt.assert_array_equal(alt.shape, self.seq.shape)
        self.assertFalse(torch.any(alt == self.seq))
        model = FixedRateSubstitutionModel(global_rate=0.)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
        npt.assert_array_equal(pos.shape, [0])
        npt.assert_array_equal(alt.shape, [0])
        model = FixedRateSubstitutionModel(global_rate=0.5)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
        self.assertAlmostEqual(alt.size(0) / self.seq_len, 0.5, 1)
        self.assertAlmostEqual(pos.size(0) / self.seq_len, 0.5, 1)
        self.assertFalse(torch.any(alt == self.seq[pos - 1]))  # convert to zero indexing
        npt.assert_array_equal(ref, self.seq[pos - 1])

        A = BASE_TO_INDEX['A']
        G = BASE_TO_INDEX['G']
        C = BASE_TO_INDEX['C']
        T = BASE_TO_INDEX['T']
        model = FixedRateSubstitutionModel(transition_rate=1.)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
        self.assertTrue(torch.all(alt[self.seq == A] == G))
        self.assertTrue(torch.all(alt[self.seq == G] == A))
        self.assertTrue(torch.all(alt[self.seq == C] == T))
        self.assertTrue(torch.all(alt[self.seq == T] == C))
        model = FixedRateSubstitutionModel(transversion_rate=1.)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
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
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0)
        self.assertAlmostEqual(torch.sum((alt == A).float()).item() / self.seq_len,
                                (0.5 + 0.5 + 0.1) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == G).float()).item() / self.seq_len,
                                (0.1 + 0.1 + 0.3) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == C).float()).item() / self.seq_len,
                                (0.8 + 0. + 0.1) * 0.25, 1)
        self.assertAlmostEqual(torch.sum((alt == T).float()).item() / self.seq_len,
                                (0.1 + 0.2 + 0.1) * 0.25, 1)
        self.assertFalse(torch.any(self.seq[pos[(alt == C)] - 1] == T))

        coords = torch.arange(self.seq_len // 2)
        model = FixedRateSubstitutionModel(global_rate=1.)
        chrom, pos, ref, alt = gen_variants(model, self.seq, 'N/A', 0, omit_indexes=coords)
        self.assertAlmostEqual(pos.size(0) / self.seq_len, 0.5, 1)
        npt.assert_array_equal(pos - 1, torch.arange(self.seq_len // 2, self.seq_len))

        variants = gen_variants(model, self.seq, 'N/A', 0,
                                omit_indexes=coords, as_pyvcf_format=True)
        for record in variants:
            self.assertEqual(record.CHROM, 'N/A')
            self.assertTrue(isinstance(record.POS, int))
            self.assertTrue(isinstance(record.REF, str))
            self.assertTrue(isinstance(record.ALT[0].sequence, str))

    def test_apply_variants(self):
        A = BASE_TO_INDEX['A']
        G = BASE_TO_INDEX['G']
        C = BASE_TO_INDEX['C']
        T = BASE_TO_INDEX['T']
        ref_seq = bioseq_to_index(self.ref_seq[self.chr][self.pos:self.pos + 25])
        var_seq = apply_variants(ref_seq, self.chr, self.pos, self.variants.fetch(self.chr))
        self.assertEqual(len(ref_seq) + 3, len(var_seq))
        npt.assert_array_equal(ref_seq[:2], var_seq[:2])
        self.assertEqual(var_seq[2], T)
        npt.assert_array_equal(ref_seq[3:6], var_seq[3:6])
        self.assertEqual(var_seq[6], G)
        npt.assert_array_equal(ref_seq[7:9], var_seq[7:9])
        npt.assert_array_equal(var_seq[9:14], [G, G, G, G, A])
        npt.assert_array_equal(ref_seq[9:11], var_seq[9+5:11+5])
        npt.assert_array_equal(ref_seq[13:19], var_seq[13+3:19+3])
        self.assertEqual(var_seq[19+3], G)
        npt.assert_array_equal(ref_seq[20:], var_seq[20 + 3:])
        
        model = FixedRateSubstitutionModel(global_rate=0.5)
        variants = gen_variants(model, self.seq, self.chr, self.pos, as_pyvcf_format=True)
        var_seq = apply_variants(self.seq, self.chr, self.pos, variants)
        n_variants = torch.sum((self.seq == var_seq).float()).item()
        self.assertAlmostEqual(n_variants / self.seq_len, 0.5, 1)

    def test_calc_mutation_rates(self):
        ts_rate, tv_rate = calc_mutation_rates(self.seq, self.seq)
        self.assertEqual(ts_rate, 0.)
        self.assertEqual(tv_rate, 0.)
        model = FixedRateSubstitutionModel(transition_rate=0.1, transversion_rate=0.3)
        variants = gen_variants(model, self.seq, self.chr, self.pos, as_pyvcf_format=True)
        var_seq = apply_variants(self.seq, self.chr, self.pos, variants)
        ts_rate, tv_rate = calc_mutation_rates(self.seq, var_seq)
        self.assertAlmostEqual(ts_rate, 0.1, 1)
        self.assertAlmostEqual(tv_rate, 0.3, 1)


if __name__ == '__main__':
    unittest.main()
