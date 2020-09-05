import sys
sys.path.append('./src')
import unittest
import numpy.testing as npt

from seqmodel.seqdata.dataset.datasets import *


class Test_DownloadableDataset(unittest.TestCase):

    def setUp(self):
        pass

    def test_cache_path(self):
        pass  #TODO


class Test_SeqIntervals(unittest.TestCase):

    def setUp(self):
        self.test_filename = 'test/data/test-regions.bed'
        self.table = pd.DataFrame(data={
            'a': ['x', 'y', 'z'],
            'b': [0, 100, 200],
            'c': [50, 150, 250],
            'd': [25, 125, 225]})
        self.intervals = SeqIntervals.from_bed_file(self.test_filename)

    def test_init(self):
        intervals = SeqIntervals(self.table)
        npt.assert_array_equal(intervals.seqnames, self.table['a'])
        npt.assert_array_equal(intervals.start, self.table['b'])
        npt.assert_array_equal(intervals.end, self.table['c'])
        npt.assert_array_equal(intervals.length, self.table['c'] - self.table['b'])
        intervals = SeqIntervals(self.table, [1, 3, 2])
        npt.assert_array_equal(intervals.seqnames, self.table['b'])
        npt.assert_array_equal(intervals.start, self.table['d'])
        npt.assert_array_equal(intervals.end, self.table['c'])
        npt.assert_array_equal(intervals.length, self.table['c'] - self.table['d'])
        intervals = SeqIntervals(self.table, ['c', 'b', 'd'])
        npt.assert_array_equal(intervals.seqnames, self.table['c'])
        npt.assert_array_equal(intervals.start, self.table['b'])
        npt.assert_array_equal(intervals.end, self.table['d'])
        npt.assert_array_equal(intervals.length, self.table['d'] - self.table['b'])

    def test_filter_by_len(self):
        intervals = SeqIntervals(self.table, ['a', 'c', 'b'], nonzero=False)
        npt.assert_array_equal(intervals.seqnames, self.table['a'])
        npt.assert_array_equal(intervals.start, self.table['c'])
        npt.assert_array_equal(intervals.end, self.table['b'])
        intervals = SeqIntervals(self.table, ['a', 'c', 'b'], nonzero=True)
        self.assertEqual(len(intervals.table), 0)
        intervals = SeqIntervals(self.table, nonzero=False)
        intervals = self.intervals.filter_by_len(min_len=50)
        self.assertTrue((intervals.length >= 50).all())
        self.assertEqual(len(intervals), 10)
        intervals = self.intervals.filter_by_len(max_len=75)
        self.assertTrue((intervals.length <= 75).all())
        self.assertEqual(len(intervals), 5)

    def test_filter(self):
        interval = self.intervals.filter('gap50')
        self.assertTrue((interval.seqnames == 'gap50').all())
        self.assertEqual(len(interval), 3)
        interval = self.intervals.filter('overlap75', 'contained20')
        self.assertTrue(((interval.seqnames == 'overlap75') | \
                    (interval.seqnames == 'contained20')).all())
        self.assertEqual(len(interval), 7)
        interval = self.intervals.filter(100, col_to_search=self.intervals.start)
        self.assertTrue((interval.start == 100).all())
        self.assertEqual(len(interval), 3)

    def test_union(self):
        interval_1 = self.intervals.filter('gap50')
        union = interval_1.union()
        npt.assert_array_equal(interval_1.table.values, union.table.values)
        npt.assert_array_equal(interval_1.table.values, union.table.values)
        union = interval_1.union(min_allowable_gap=51)
        self.assertEqual(len(union), 1)
        self.assertEqual(union.start[0], 0)
        self.assertEqual(union.length[0], 400)
        interval_2 = self.intervals.filter('contained20')
        union = interval_2.union()
        self.assertEqual(len(union), 1)
        self.assertEqual(union.start.iloc[0], 0)
        self.assertEqual(union.length.iloc[0], 300)
        interval_3 = self.intervals.filter('adjacent100')
        union = interval_1.union(interval_2, interval_3, min_allowable_gap=51)
        self.assertEqual(len(union), 3)
        union = self.intervals.union()
        self.assertEqual(len(union), 6)
        self.assertEqual(union.filter('overlap75').start.iloc[0], 100)
        self.assertEqual(union.filter('overlap75').end.iloc[0], 275)

    def test_intersect(self):
        pass  #TODO

    def test_remove(self):
        pass  #TODO


if __name__ == '__main__':
    unittest.main()
