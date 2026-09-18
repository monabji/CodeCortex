from __future__ import annotations
import csv
import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mutantscope.esm_features import assemble_features
from mutantscope.esm_features import file_hash
from mutantscope.primary_model import (PrimaryMLP, TrainingConfig, fit_normalizer,
    load_checkpoint, predict, save_checkpoint, selected_rows, write_predictions)


class PrimaryModelTests(unittest.TestCase):
    def test_saved_prediction_metrics_preserve_float32_reduction(self):
        script = Path(__file__).resolve().parents[1] / "scripts/verify_phase4_artifacts.py"
        module_spec = importlib.util.spec_from_file_location("phase4_verifier", script)
        verifier = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(verifier)
        from mutantscope.baseline import regression_metrics
        rows = [{"source_row": str(i), "wild_type_name": "fixture", "cluster": "fixture",
                 "mutation": "A1V"} for i in range(17)]
        observed = np.random.default_rng(2).normal(size=17)
        predictions = np.random.default_rng(5).normal(size=17).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv.gz"
            write_predictions(path, rows, observed, predictions)
            actual = verifier.verify_predictions(path, rows, observed, predictions)
            self.assertEqual(actual, regression_metrics(observed, predictions))

    def test_streaming_normalizer_matches_direct_population_statistics(self):
        base = np.random.default_rng(5).normal(size=(17, 4, 480)).astype(np.float32)
        mean, scale = fit_normalizer(base, 3)
        full = assemble_features(base).astype(np.float64)
        np.testing.assert_allclose(mean, full.mean(axis=0), atol=1e-7)
        np.testing.assert_allclose(scale, full.std(axis=0), atol=1e-7)
        with self.assertRaises(ValueError):
            fit_normalizer(base[:0])

    def test_checkpoint_roundtrip_preserves_prediction_and_train_only_normalizer(self):
        torch.manual_seed(1)
        config = TrainingConfig(hidden_sizes=(8, 4))
        model = PrimaryMLP(config)
        train = np.random.default_rng(3).normal(size=(8, 4, 480)).astype(np.float32)
        validation = train + 10
        mean, scale = fit_normalizer(train)
        expected = predict(model, validation, mean, scale)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            from dataclasses import asdict
            save_checkpoint(path, {"config": asdict(config), "model_state": model.state_dict(),
                "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale)})
            _, restored, saved_mean, saved_scale = load_checkpoint(path)
            np.testing.assert_array_equal(expected, predict(restored, validation, saved_mean, saved_scale))
            np.testing.assert_array_equal(saved_mean, mean)

    def test_development_loader_skips_invalid_test_targets_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv.gz"
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("source_row", "split", "ddg_kcal_mol"))
                writer.writeheader()
                writer.writerows([{"source_row": 1, "split": "train", "ddg_kcal_mol": "0.2"},
                    {"source_row": "invalid", "split": "test", "ddg_kcal_mol": "invalid"}])
            self.assertEqual(len(selected_rows(path, ("train",))["train"]), 1)
            with self.assertRaises(ValueError):
                selected_rows(path, ("test",))
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("source_row", "split", "ddg_kcal_mol"))
                writer.writeheader()
                writer.writerows([{"source_row": 1, "split": "train", "ddg_kcal_mol": "0.2"}] * 2)
            with self.assertRaises(ValueError):
                selected_rows(path, ("train",))

    def test_frozen_test_gate_rejects_smoke_changes_and_repeated_evaluation(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_phase4_test.py"
        module_spec = importlib.util.spec_from_file_location("phase4_evaluator", script)
        evaluator = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(evaluator)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            # These are opaque files: the freeze gate must hash, not parse test content.
            for filename in ("records.gz", "summary.json", "baseline.npz", "best_model.pt"):
                (output / filename).write_bytes(b"opaque test fixture")
            selection = {"status": "smoke_only", "test_set_accessed": False,
                "source_records_sha256": file_hash(output / "records.gz"),
                "split_summary_sha256": file_hash(output / "summary.json"),
                "baseline_model_sha256": file_hash(output / "baseline.npz"),
                "checkpoint_sha256": file_hash(output / "best_model.pt")}
            path = output / "selection.json"
            path.write_text(json.dumps(selection))
            arguments = (output, output / "records.gz", output / "summary.json", output / "baseline.npz")
            with self.assertRaises(ValueError):
                evaluator.frozen_contract(*arguments)
            self.assertFalse((output / "test_evaluation_guard.json").exists())
            selection["status"] = "frozen"
            path.write_text(json.dumps(selection))
            (output / "development_verification.json").write_text(json.dumps({"status": "verified",
                "checkpoint_sha256": selection["checkpoint_sha256"], "normalization_verified_train_only": True,
                "exact_development_feature_join_verified": True}))
            _, contract = evaluator.frozen_contract(*arguments)
            evaluator.frozen_contract(*arguments)  # Interrupted extraction may resume the same contract.
            selection["selected_epoch"] = 2
            path.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                evaluator.frozen_contract(*arguments)
            selection.pop("selected_epoch")
            path.write_text(json.dumps(selection))
            (output / "test_evaluation_guard.json").write_text(json.dumps({"status": "evaluated", "contract": contract}))
            with self.assertRaises(FileExistsError):
                evaluator.frozen_contract(*arguments)


if __name__ == "__main__":
    unittest.main()
