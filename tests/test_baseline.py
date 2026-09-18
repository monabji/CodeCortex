from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mutantscope.baseline import FEATURE_NAMES, feature_matrix, fit_ridge, regression_metrics


def row(wild_type: str, mutant: str, position: int, sequence: str) -> dict[str, str]:
    return {"wild_type_residue": wild_type, "mutant_residue": mutant, "position": str(position), "wild_type_sequence": sequence}


class BaselineTests(unittest.TestCase):
    def test_feature_schema_is_inference_available_and_stable(self) -> None:
        values = feature_matrix([row("A", "V", 1, "ACD")])
        self.assertEqual(values.shape, (1, len(FEATURE_NAMES)))
        self.assertEqual(values[0, 0], 1.0)
        self.assertEqual(values[0, 20 + 17], 1.0)
        self.assertAlmostEqual(values[0, 40], 1 / 3)
        self.assertEqual(values[0, 41], 3.0)

    def test_ridge_uses_train_statistics_and_returns_finite_predictions(self) -> None:
        train = feature_matrix([row("A", "V", 1, "ACD"), row("D", "E", 2, "ACD"), row("L", "I", 3, "ACD")])
        labels = np.asarray([1.0, -1.0, 0.5])
        model = fit_ridge(train, labels, 0.1)
        prediction = model.predict(feature_matrix([row("A", "V", 1, "ACD")]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.standardizer.mean.shape[0], len(FEATURE_NAMES))

    def test_metrics_report_expected_perfect_fit(self) -> None:
        observed = np.asarray([-1.0, 0.0, 2.0])
        metrics = regression_metrics(observed, observed.copy())
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["pearson"], 1.0)
        self.assertEqual(metrics["spearman"], 1.0)
