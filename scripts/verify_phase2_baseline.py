"""Independently verify Phase 2 validation-only baseline artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.baseline import FEATURE_NAMES, regression_metrics  # noqa: E402


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase2")
    args = parser.parse_args()
    artifacts = args.artifacts_dir
    metrics = json.loads((artifacts / "validation_metrics.json").read_text(encoding="utf-8"))
    if metrics["test_set_accessed"] is not False:
        raise AssertionError("Phase 2 artifact incorrectly claims test-set access")
    if sha256sum(args.records) != metrics["source_records_sha256"]:
        raise AssertionError("Phase 1 records hash does not match Phase 2 provenance")

    validation_source_rows: set[int] = set()
    with gzip.open(args.records, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "validation":
                validation_source_rows.add(int(row["source_row"]))
    prediction_source_rows: set[int] = set()
    observed, predicted = [], []
    with gzip.open(artifacts / "validation_predictions.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_row = int(row["source_row"])
            if source_row not in validation_source_rows or source_row in prediction_source_rows:
                raise AssertionError(f"Prediction is not a unique validation record: {source_row}")
            prediction_source_rows.add(source_row)
            observed.append(float(row["observed_ddg_kcal_mol"]))
            predicted.append(float(row["predicted_ddg_kcal_mol"]))
    if prediction_source_rows != validation_source_rows:
        raise AssertionError("Predictions do not cover exactly the validation split")
    recomputed = regression_metrics(np.asarray(observed), np.asarray(predicted))
    for metric in ("mae", "rmse", "pearson", "spearman"):
        if not np.isclose(recomputed[metric], metrics["selected_validation_metrics"][metric]):
            raise AssertionError(f"Validation {metric} differs from recorded result")
    model = np.load(artifacts / "ridge_baseline_model.npz", allow_pickle=False)
    if tuple(model["feature_names"].tolist()) != FEATURE_NAMES:
        raise AssertionError("Model feature schema differs from recorded baseline schema")
    if len(model["weights"]) != len(FEATURE_NAMES):
        raise AssertionError("Model weight dimension is invalid")
    print(json.dumps({"validation_predictions": len(prediction_source_rows), "metrics": recomputed, "test_set_accessed": metrics["test_set_accessed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
