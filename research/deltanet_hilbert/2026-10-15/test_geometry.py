import unittest
from fractions import Fraction

from geometry import constant_recall, gram, mean_identity, order_and_drift, rank, read, write


class ExactGeometryTests(unittest.TestCase):
    def test_simplex_cannot_store_constant_targets(self):
        rows = constant_recall()
        row = next(r for r in rows if r['period'] == 3 and r['geometry'] == 'simplex')
        self.assertEqual([Fraction(x) for x in row['readouts_exact']],
                         [Fraction(-7, 8), Fraction(-1, 8), Fraction(1)])
        self.assertEqual(Fraction(row['mean_squared_error_exact']), Fraction(51, 32))

    def test_orthogonal_write_preserves_other_address(self):
        g = gram(3, 'orthogonal')
        state = [Fraction(0)] * 3
        write(state, g, 0, Fraction(2), Fraction(1))
        write(state, g, 1, Fraction(-3), Fraction(1))
        self.assertEqual(read(state, g), [Fraction(2), Fraction(-3), Fraction(0)])

    def test_exact_rank_and_mean(self):
        self.assertEqual(rank(gram(3, 'simplex')), 2)
        self.assertEqual(rank(gram(3, 'orthogonal')), 3)
        self.assertTrue(mean_identity()['every_prefix_exact'])
        self.assertTrue(mean_identity()['additive_counts_exact'])

    def test_change_point_and_order_are_distinct_from_mean(self):
        result = order_and_drift()
        self.assertEqual(result['change_point_mean_exact'], '1/5')
        self.assertEqual(result['change_point_fixed_exact'], '175/256')
        self.assertEqual(result['limiting_independent_sample_equivalent_exact'], '7')


if __name__ == '__main__':
    unittest.main()
