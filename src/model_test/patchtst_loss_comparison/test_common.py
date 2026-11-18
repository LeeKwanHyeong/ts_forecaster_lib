"""시간/지표 계약을 위한 외부 서비스 없는 단위 테스트."""

import unittest
from datetime import date

import numpy as np

from model_test.patchtst_loss_comparison.common import calendar_features, scores, week_date, week_id


class ComparisonContracts(unittest.TestCase):
    def test_metrics_in_original_units(self):
        result = scores([1, 3], [2, 1])
        self.assertEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))
        self.assertEqual(result["wape_pct"], 75.0)
        self.assertEqual(result["bias_pct"], -25.0)

    def test_zero_demand_has_no_percentage_denominator(self):
        self.assertIsNone(scores([0, 0], [1, 2])["wape_pct"])

    def test_nonfinite_and_wrong_shapes_rejected(self):
        with self.assertRaises(ValueError):
            scores([1], [float("nan")])
        with self.assertRaises(ValueError):
            scores([1], [[1]])

    def test_iso_week_53_and_year_boundary(self):
        self.assertEqual(week_date(202053), date(2020, 12, 28))
        self.assertEqual(week_id(date(2021, 1, 4)), 202101)
        self.assertEqual((week_date(202101) - week_date(202053)).days, 7)

    def test_features_do_not_depend_on_snapshot_length(self):
        first = calendar_features(np.array([202052, 202053]))
        later = calendar_features(np.array([202052, 202053, 202101]))
        np.testing.assert_array_equal(first, later[:2])


if __name__ == "__main__":
    unittest.main()
