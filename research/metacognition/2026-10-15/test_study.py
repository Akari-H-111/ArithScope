"""Independent small cases and failure checks for the branch's exact oracle."""
import unittest
from fractions import Fraction

from study import correlated_copies, enumerate_risks, likelihood_posterior, posterior, share_collision


class ExactChecks(unittest.TestCase):
    def test_original_task_one_observation(self):
        self.assertEqual(posterior((1,) + (0,) * 15), (Fraction(17, 32),) + (Fraction(1, 32),) * 15)

    def test_original_task_collision_known_value(self):
        result = share_collision()
        self.assertEqual(Fraction(result["small_top_p_exact"]), Fraction(289, 320))
        self.assertGreater(Fraction(result["common_prediction_minimum_mean_brier_regret_exact"]), 0)

    def test_exchangeability_and_uniform_tie(self):
        self.assertEqual(likelihood_posterior((0, 1, 0), 3, Fraction(1, 2)), posterior((2, 1, 0)))
        self.assertEqual(posterior((3, 3, 3)), (Fraction(1, 3),) * 3)

    def test_reject_invalid_count_inputs(self):
        for counts in ((1,), (1, -1), (1.0, 0), (True, 0)):
            with self.assertRaises(ValueError):
                posterior(counts)
        with self.assertRaises(ValueError):
            posterior((1, 0), Fraction(0))

    def test_direct_binary_bayes_risk(self):
        rows = enumerate_risks(vocab=2, max_length=1)
        self.assertEqual(Fraction(rows[0]["bayes_brier_exact"]), Fraction(1, 2))
        self.assertEqual(Fraction(rows[1]["bayes_brier_exact"]), Fraction(3, 8))

    def test_duplicate_law_has_fixed_information(self):
        rows = correlated_copies()
        self.assertTrue(all(row["proper_brier_exact"] == "1/2" for row in rows))
        self.assertTrue(all(row["proper_nll"] == rows[0]["proper_nll"] for row in rows))
        self.assertEqual(Fraction(rows[0]["excess_brier_exact"]), 0)
        self.assertGreater(Fraction(rows[-1]["excess_brier_exact"]), 0)


if __name__ == "__main__":
    unittest.main()
