"""Reproduce Phase 4 statistics and saved predictions without changing selection."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mutantscope.baseline import FEATURE_NAMES, RidgeModel, Standardizer, feature_matrix, regression_metrics, targets
from mutantscope.esm_features import EncoderSpec, FeatureRecord, FrozenEsmEncoder, file_hash, load_shard, write_json_atomic
from mutantscope.primary_model import fit_normalizer, load_checkpoint, predict, selected_rows


def verify_predictions(path, rows, observed, reproduced):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(rows):
        raise AssertionError("Saved prediction coverage differs")
    estimates = []
    for record, row, label in zip(saved, rows, observed, strict=True):
        if any(record[key] != row[key] for key in ("source_row", "wild_type_name", "cluster", "mutation")):
            raise AssertionError("Prediction identifiers/partition changed")
        if float(record["observed_ddg_kcal_mol"]) != label:
            raise AssertionError("Saved target changed")
        estimate = float(record["predicted_ddg_kcal_mol"])
        np.testing.assert_allclose(float(record["residual_kcal_mol"]), estimate - label, atol=1e-12, rtol=0)
        estimates.append(estimate)
    np.testing.assert_allclose(estimates, reproduced, atol=1e-6, rtol=1e-6)
    # CSV parses as float64; preserve the original predictor's dtype for the
    # metric reduction (MLP float32, ridge float64) rather than changing it.
    return regression_metrics(observed, np.asarray(estimates, dtype=np.asarray(reproduced).dtype))


def assert_metrics(actual, expected):
    if actual["count"] != expected["count"]:
        raise AssertionError("Metric count differs")
    for key in ("mae", "rmse", "pearson", "spearman"):
        if actual[key] is None or expected[key] is None:
            if actual[key] != expected[key]:
                raise AssertionError("Undefined correlation differs")
        else:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-10)


def verify_grouped(path, rows, observed, predictions, key):
    with path.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    names = sorted({row[key] for row in rows})
    if [record[key] for record in saved] != names:
        raise AssertionError("Grouped diagnostic coverage changed")
    maes = []
    for record, name in zip(saved, names, strict=True):
        indexes = [i for i, row in enumerate(rows) if row[key] == name]
        actual = regression_metrics(observed[indexes], predictions[indexes])
        expected = {field: int(record[field]) if field == "count" else
                    (float(record[field]) if record[field] else None) for field in actual}
        assert_metrics(actual, expected)
        maes.append(actual["mae"])
    return {"count": len(names), "macro_mae": float(np.mean(maes)),
            "min_mae": min(maes), "max_mae": max(maes)}


def verify(args):
    output = args.artifacts_dir
    selection = json.loads((output / "selection.json").read_text())
    summary = json.loads(args.split_summary.read_text())
    if selection["status"] != "frozen" or selection["test_set_accessed"] is not False:
        raise AssertionError("Full development-only frozen selection required")
    for path, key in ((args.records, "source_records_sha256"), (args.split_summary, "split_summary_sha256"),
        (args.phase3_cache / "manifest.json", "phase3_manifest_sha256"),
        (output / "best_model.pt", "checkpoint_sha256"), (args.baseline_model, "baseline_model_sha256"),
        (output / "validation_predictions.csv.gz", "validation_predictions_sha256")):
        if file_hash(path) != selection[key]:
            raise AssertionError(f"Changed provenance: {path.name}")
    checkpoint, model, mean, scale = load_checkpoint(output / "best_model.pt")
    if json.loads(json.dumps(checkpoint["config"])) != selection["config"]:
        raise AssertionError("Checkpoint configuration differs from selection")
    for name, checksum in selection["code_sha256"].items():
        if file_hash(ROOT / name) != checksum:
            raise AssertionError("Recorded training implementation changed")
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    if (checkpoint["smoke_rows"] or checkpoint["test_set_accessed"] is not False
            or checkpoint["normalizer_fitted_split"] != "train"
            or checkpoint["normalizer_count"] != summary["splits"]["train"]["records"]):
        raise AssertionError("Training/checkpoint discipline changed")
    if checkpoint["provenance"] != {key: selection[key] for key in checkpoint["provenance"]}:
        raise AssertionError("Checkpoint/selection mismatch")
    history = json.loads((output / "training_history.json").read_text())
    if history["test_set_accessed"] is not False:
        raise AssertionError("History indicates test access")
    best = min(history["epochs"], key=lambda epoch: (epoch["validation_metrics"]["mae"], epoch["epoch"]))
    if best["epoch"] != checkpoint["epoch"] or checkpoint["epoch"] != selection["selected_epoch"]:
        raise AssertionError("Checkpoint is not minimum validation MAE")
    prepared = json.loads((args.prepared_dir / "manifest.json").read_text())
    partitions = selected_rows(args.records, ("train", "validation"))
    arrays = {}
    phase3 = json.loads((args.phase3_cache / "manifest.json").read_text())
    spec = EncoderSpec(**selection["encoder"])
    if spec != EncoderSpec():
        raise AssertionError("Unexpected encoder contract")
    for split, rows in partitions.items():
        path = args.prepared_dir / f"{split}_base.npy"
        if file_hash(path) != prepared["base_files_sha256"][split]:
            raise AssertionError("Prepared matrix checksum changed")
        arrays[split] = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(rows) != summary["splits"][split]["records"] or len(arrays[split]) != len(rows):
            raise AssertionError("Full data coverage mismatch")
        offsets = {int(row["source_row"]): i for i, row in enumerate(rows)}
        seen = set()
        for entry in phase3["shards"]:
            if entry["split"] != split:
                continue
            path = args.phase3_cache / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise AssertionError("Changed Phase 3 shard")
            base, identifiers, _ = load_shard(path, spec)
            indexes = [offsets[int(identifier)] for identifier in identifiers]
            load_shard(path, spec, [FeatureRecord.from_row(rows[index]) for index in indexes])
            if seen.intersection(identifiers.tolist()):
                raise AssertionError("Duplicate feature coverage")
            seen.update(identifiers.tolist())
            np.testing.assert_array_equal(base, arrays[split][indexes])
        if seen != set(offsets):
            raise AssertionError("Prepared base vectors lack exact coverage")
    fitted_mean, fitted_scale = fit_normalizer(arrays["train"])
    np.testing.assert_array_equal(mean, fitted_mean)
    np.testing.assert_array_equal(scale, fitted_scale)
    validation_metrics = verify_predictions(output / "validation_predictions.csv.gz", partitions["validation"],
        targets(partitions["validation"]), predict(model, arrays["validation"], mean, scale, checkpoint["config"]["batch_size"]))
    assert_metrics(validation_metrics, selection["validation_metrics"])
    assert_metrics(validation_metrics, checkpoint["validation_metrics"])
    report = {"status": "verified", "normalization_verified_train_only": True,
        "exact_development_feature_join_verified": True, "validation_metrics": validation_metrics,
        "selected_epoch": checkpoint["epoch"], "checkpoint_sha256": file_hash(output / "best_model.pt"),
        "development_only": args.development_only, "test_artifacts_verified": False}
    if not args.development_only:
        guard = json.loads((output / "test_evaluation_guard.json").read_text())
        evaluation = json.loads((output / "test_metrics.json").read_text())
        if (guard["status"] != "evaluated" or guard["contract"] != evaluation["contract"]
                or guard["test_metrics_sha256"] != file_hash(output / "test_metrics.json")):
            raise AssertionError("Incomplete/changed frozen evaluation")
        for path, key in ((output / "selection.json", "selection_sha256"),
            (output / "best_model.pt", "checkpoint_sha256"), (args.baseline_model, "baseline_model_sha256"),
            (args.records, "source_records_sha256"), (args.split_summary, "split_summary_sha256")):
            if file_hash(path) != evaluation["contract"][key]:
                raise AssertionError("Test contract changed")
        rows = selected_rows(args.records, ("test",))["test"]
        manifest = json.loads((args.test_cache_dir / "manifest.json").read_text())
        if (manifest["status"] != "complete" or manifest["contract"] != guard["contract"]
                or manifest["labels_used"] is not False or len(rows) != summary["splits"]["test"]["records"]
                or file_hash(args.test_cache_dir / "manifest.json") != evaluation["test_cache_manifest_sha256"]
                or file_hash(args.test_cache_dir / "test_base.npy") != manifest["base_sha256"]):
            raise AssertionError("Test cache provenance/coverage mismatch")
        base = np.load(args.test_cache_dir / "test_base.npy", mmap_mode="r", allow_pickle=False)
        offset = 0
        for entry in manifest["shards"]:
            path = args.test_cache_dir / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise AssertionError("Changed test shard")
            matching = [replace(FeatureRecord.from_row({**row, "split": "validation"}), split="test")
                        for row in rows[offset:offset + entry["records"]]]
            values, _, _ = load_shard(path, spec, matching)
            np.testing.assert_array_equal(values, base[offset:offset + len(values)])
            offset += len(values)
        if offset != len(rows) or len(base) != len(rows):
            raise AssertionError("Incomplete test features")
        observed = targets(rows)
        primary_predictions = predict(model, base, mean, scale, checkpoint["config"]["batch_size"])
        primary = verify_predictions(output / "test_primary_predictions.csv.gz", rows, observed,
                                     primary_predictions)
        with np.load(args.baseline_model, allow_pickle=False) as saved:
            if saved["feature_names"].tolist() != list(FEATURE_NAMES):
                raise AssertionError("Baseline schema changed")
            baseline = RidgeModel(float(saved["alpha"][0]), float(saved["intercept"][0]), saved["weights"],
                Standardizer(saved["feature_mean"], saved["feature_scale"]))
        baseline_predictions = baseline.predict(feature_matrix(rows))
        baseline_metrics = verify_predictions(output / "test_baseline_predictions.csv.gz", rows, observed, baseline_predictions)
        assert_metrics(primary, evaluation["primary_metrics"])
        assert_metrics(baseline_metrics, evaluation["baseline_metrics"])
        for name, checksum in evaluation["prediction_artifacts_sha256"].items():
            if file_hash(output / name) != checksum:
                raise AssertionError("Prediction/diagnostic artifact changed")
        for name, predictions in (("primary", primary_predictions), ("baseline", baseline_predictions)):
            for key, suffix in (("wild_type_name", "protein"), ("cluster", "cluster")):
                actual = verify_grouped(output / f"test_{name}_per_{suffix}_metrics.csv", rows, observed, predictions, key)
                expected = evaluation["grouped_metrics"][f"{name}_{suffix}"]
                if actual["count"] != expected["count"]:
                    raise AssertionError("Grouped summary count differs")
                for field in ("macro_mae", "min_mae", "max_mae"):
                    np.testing.assert_allclose(actual[field], expected[field], atol=1e-10, rtol=1e-10)
        if evaluation["counts"] != summary["splits"]["test"]:
            raise AssertionError("Test protein/cluster counts changed")
        with (output / "test_sample.csv").open(newline="") as stream:
            examples = list(csv.DictReader(stream))
        indexes = {row["source_row"]: i for i, row in enumerate(rows)}
        if len(examples) != 5 or len({example["source_row"] for example in examples}) != 5:
            raise AssertionError("Invalid human-checkable sample")
        for example in examples:
            i = indexes[example["source_row"]]
            if example["mutation"] != rows[i]["mutation"]:
                raise AssertionError("Sample mutation differs")
            np.testing.assert_array_equal([float(example[field]) for field in ("observed_ddg", "primary_ddg", "baseline_ddg")],
                                          [observed[i], primary_predictions[i], baseline_predictions[i]])
        encoder = FrozenEsmEncoder(spec, "cpu", True, 16)
        sample_indexes = set(np.random.default_rng(20260918).choice(len(rows), size=4, replace=False).tolist())
        for predicate in (lambda row: int(row["position"]) == 1,
                          lambda row: int(row["position"]) == len(row["wild_type_sequence"])):
            index = next((i for i, row in enumerate(rows) if predicate(row)), None)
            if index is not None:
                sample_indexes.add(index)
        ordered = sorted(sample_indexes)
        wilds = encoder.encode([rows[i]["wild_type_sequence"] for i in ordered])
        mutants = encoder.encode([rows[i]["mutant_sequence"] for i in ordered])
        maximum_error = 0.0
        for i, wild, mutant in zip(ordered, wilds, mutants, strict=True):
            position = int(rows[i]["position"]) - 1
            fresh = np.stack((wild[position], mutant[position], wild.mean(axis=0), mutant.mean(axis=0)))
            np.testing.assert_allclose(base[i], fresh, atol=3e-5, rtol=3e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(base[i] - fresh))))
        if encoder.model.training or any(parameter.requires_grad or parameter.grad is not None
                                         for parameter in encoder.model.parameters()):
            raise AssertionError("Test verifier encoder not frozen")
        if (output / "test_diagnostics.png").stat().st_size < 1000:
            raise AssertionError("Diagnostic plot missing/empty")
        report.update(test_artifacts_verified=True, primary_test_metrics=primary, baseline_test_metrics=baseline_metrics)
        report.update(grouped_diagnostics_verified=True, test_sample_verified=True,
            live_test_reencoding={"records": len(ordered), "max_absolute_error": maximum_error,
                                 "encoder_frozen": True},
            diagnostic_plot_sha256=file_hash(output / "test_diagnostics.png"),
            verification_code_sha256=file_hash(Path(__file__).resolve()))
    write_json_atomic(output / ("development_verification.json" if args.development_only else "verification_report.json"), report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--phase3-cache", type=Path, default=ROOT / "data/cache/phase3/esm2_site_global_v1")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "data/cache/phase4/development_v1")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "artifacts/phase2/ridge_baseline_model.npz")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase4")
    parser.add_argument("--test-cache-dir", type=Path, default=ROOT / "data/cache/phase4/test_v1")
    parser.add_argument("--development-only", action="store_true")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
