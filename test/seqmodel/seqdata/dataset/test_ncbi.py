import sys
sys.path.append('./src')
import os
import unittest
import numpy.testing as npt

from seqmodel.seqdata.dataset.ncbi import *
from seqmodel.seqdata.dataset.datasets import decompressed_name


class Test_NCBIDataset(unittest.TestCase):

    _DO_DOWNLOAD = False

    def setUp(self):
        self.cache_root = 'test/data/seqdata/dataset/.cache'
        self.test_name = 'ASM308665v1'  # use yeast sequence, as it is short
        self.test_accn = 'GCA_003086655.1'
        self.seq = NCBIDataset.from_name(self.test_name, self.cache_root,
                                    download=self._DO_DOWNLOAD)

    def del_if_exists(self, *paths):
        for path in paths:
            fullpath = os.path.abspath(os.path.join(
                self.cache_root, 'NCBIDataset', self.test_accn + '_' + self.test_name, path))
            if os.path.exists(fullpath):
                if os.path.isfile(fullpath):
                    os.remove(fullpath)
                if os.path.isdir(fullpath):
                    os.rmdir(fullpath)

    def test_from_name(self):
        seq = NCBIDataset.from_name(self.test_accn, self.cache_root, download=False)
        self.assertEqual(self.seq.root_name, self.test_accn + '_' + self.test_name)
        self.assertEqual(self.seq.cache_root, os.path.join(
            self.cache_root, 'NCBIDataset', self.test_accn + '_' + self.test_name))
        self.assertEqual(self.seq.url_root, seq.url_root)

    def test_retrieve_or_download(self):
        md5_file = 'md5checksums.txt'
        test_download_file = self.seq.root_name + '_feature_count.txt.gz'
        test_extract_file = self.seq.root_name + '_feature_count.txt'
        if self._DO_DOWNLOAD:
            self.del_if_exists(md5_file, test_download_file,
                        test_extract_file, self.cache_root)
            try:
                file = self.seq.md5_retrieve_or_download(test_download_file, decompress=False)
            except FileNotFoundError as e:
                self.assertEqual(str(e), 'Unable to retrieve or download ' + md5_file)
            self.seq.download = True
        else:
            self.del_if_exists(test_extract_file)
        file = self.seq.md5_retrieve_or_download(test_download_file, decompress=False)
        self.assertTrue(self.seq._check_exists(test_download_file))
        self.assertFalse(self.seq._check_exists(test_extract_file))
        file = self.seq.md5_retrieve_or_download(test_download_file, decompress=True)
        self.assertTrue(self.seq._check_exists(test_extract_file))

    def test_generate_faidx_from_assembly_report(self):
        pass  #FIXME: this function is currently broken?

    def test_get_tables(self):
        table = self.seq.report_table
        npt.assert_array_equal(table.columns, ['Sequence-Name', 'Sequence-Role',
            'Assigned-Molecule', 'Assigned-Molecule-Location/Type', 'GenBank-Accn',
            'Relationship', 'RefSeq-Accn', 'Assembly-Unit', 'Sequence-Length', 'UCSC-style-name'])
        self.assertGreater(len(table), 0)

    def test_fasta(self):
        file = self.seq.fasta_file
        self.assertEqual(file, os.path.join(self.cache_root, 'NCBIDataset',
                self.test_accn + '_' + self.test_name,
                self.test_accn + '_' + self.test_name + '_genomic.fna'))
        fasta = self.seq.fasta
        regions = list(fasta.keys())
        self.assertEqual(len(regions), len(self.seq.report_table))
        # note: in this case the order of sequences is the same as the report table
        # this may not always be the case
        npt.assert_array_equal(regions, list(self.seq.report_table['GenBank-Accn']))
        self.assertEqual(fasta[regions[0]][:80],
            'CCCACACAccccacacaccacacacaccacacccacacccacacacaccacaccacacccacacccacaccacacccaca')
        self.assertEqual(fasta[regions[-1]][-80:],
            'ggtgtgtgggtgtggtgtgggtgtggtgtggtgtgggtgtgggtgtggtgtgggtgtggtgtggtgtggtgtgggtgtgg')

    def test_translate_accn(self):
        pass  #TODO


if __name__ == '__main__':
    # download external data only if this file is run directly
    print('Downloading data for tests...')
    Test_NCBIDataset._DO_DOWNLOAD = True
    unittest.main()
