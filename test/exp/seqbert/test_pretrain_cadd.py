import sys
sys.path.append('./src')
import unittest
import torch
import torch.nn as nn
import numpy.testing as npt
import pyfaidx
import vcf

from seqmodel.seqdata.mapseq import random_bioseq
from seqmodel.seqdata.iterseq import StridedSequence
from seqmodel.functional import bioseq_to_index
from exp.seqbert import TOKENS_BP_IDX
from exp.seqbert.model import SeqBERT
from exp.seqbert.pretrain_cadd import *


class Test_PretrainCADD(unittest.TestCase):

    def setUp(self):
        self.seq_len = 100
        self.seq = random_bioseq(self.seq_len)
        self.metadata = ('random', 10)

    def tearDown(self):
        pass

    def test_cls_test_transform(self):
        pass #TODO function got moved into model class
        # output = cls_test_transform(self.seq)
        #TODO looks correct

    def test_variant_transform(self):
        chrom = 'chr1'
        start = 10000
        end = start + 1000
        fasta_file = 'test/data/grch38_excerpt.fa'
        vcf_file = 'test/data/sample.vcf.gz'
        ref_seq = pyfaidx.Fasta(fasta_file, as_raw=True)[chrom][start:end]
        var_seq = pyfaidx.FastaVariant(fasta_file, vcf_file, as_raw=True)[chrom][start:end]
        var_records = vcf.Reader(filename=vcf_file).fetch(chrom, start, end)
        # output = variant_transform((ref_seq, var_seq, var_records, 'sample'),
        #                             (chrom, start))
        #TODO function got moved into model class
        #print(output)  # FIXME breaks when no random variants generated


if __name__ == '__main__':
    unittest.main()
