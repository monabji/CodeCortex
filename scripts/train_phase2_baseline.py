"""Train and select the Phase 2 ridge baseline using train/validation only."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.baseline import FEATURE_NAMES, RidgeModel, feature_matrix, fit_ridge, regression_metrics, targets  # noqa: E402


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                rows.append(row)
    return rows


def per_protein_metrics(rows: list[dict[str, str]], observed: np.ndarray, predicted: np.ndarray) -> list[dict[str, object]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["wild_type_name"]].append(index)
    output: list[dict[str, object]] = []
    for protein, indexes in sorted(groups.items()):
        indices = np.asarray(indexes, dtype=np.int64)
        metrics = regression_metrics(observed[indices], predicted[indices])
        output.append({"wild_type_name": protein, **metrics})
    return output


def save_model(path: Path, model: RidgeModel) -> None:
    np.savez_compressed(
        path,
        alpha=np.asarray([model.alpha]),
        intercept=np.asarray([model.intercept]),
        weights=model.weights,
        feature_mean=model.standardizer.mean,
        feature_scale=model.standardizer.scale,
        feature_names=np.asarray(FEATURE_NAMES),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--force", action="store_true", help="Replace existing local Phase 2 artifacts")
    args = parser.parse_args()

    artifacts = args.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model": artifacts / "ridge_baseline_model.npz",
        "metrics": artifacts / "validation_metrics.json",
        "predictions": artifacts / "validation_predictions.csv.gz",
        "per_protein": artifacts / "validation_per_protein_metrics.csv",
        "sample": artifacts / "validation_sample.csv",
    }
    if not args.force and any(path.exists() for path in outputs.values()):
        existing = ", ".join(str(path) for path in outputs.values() if path.exists())
        raise FileExistsError(f"Refusing to overwrite baseline artifacts without --force: {existing}")

    split_summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    train_rows = load_rows(args.records, "train")
    validation_rows = load_rows(args.records, "validation")
    if len(train_rows) != split_summary["splits"]["train"]["records"] or len(validation_rows) != split_summary["splits"]["validation"]["records"]:
        raise ValueError("Phase 1 record counts do not match the split summary")

    train_features, validation_features = feature_matrix(train_rows), feature_matrix(validation_rows)
    train_targets, validation_targets = targets(train_rows), targets(validation_rows)
    mean_predictions = np.full_like(validation_targets, train_targets.mean())
    candidate_alphas = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
    candidates: dict[str, dict[str, float | int | None]] = {}
    fitted_models: dict[float, RidgeModel] = {}
    for alpha in candidate_alphas:
        model = fit_ridge(train_features, train_targets, alpha)
        fitted_models[alpha] = model
        candidates[str(alpha)] = regression_metrics(validation_targets, model.predict(validation_features))
    selected_alpha = min(candidate_alphas, key=lambda alpha: (float(candidates[str(alpha)]["mae"]), alpha))
    selected_model = fitted_models[selected_alpha]
    validation_predictions = selected_model.predict(validation_features)

    with gzip.open(outputs["predictions"], "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_row", "wild_type_name", "cluster", "mutation", "observed_ddg_kcal_mol", "predicted_ddg_kcal_mol", "residual_kcal_mol"))
        writer.writeheader()
        for row, observed, predicted in zip(validation_rows, validation_targets, validation_predictions, strict=True):
            writer.writerow({
                "source_row": row["source_row"], "wild_type_name": row["wild_type_name"], "cluster": row["cluster"], "mutation": row["mutation"],
                "observed_ddg_kcal_mol": format(observed, ".17g"), "predicted_ddg_kcal_mol": format(predicted, ".17g"), "residual_kcal_mol": format(predicted - observed, ".17g"),
            })
    per_protein = per_protein_metrics(validation_rows, validation_targets, validation_predictions)
    with outputs["per_protein"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("wild_type_name", "count", "mae", "rmse", "pearson", "spearman"))
        writer.writeheader()
        writer.writerows(per_protein)
    sample_indexes = np.sort(np.random.default_rng(args.seed).choice(len(validation_rows), size=5, replace=False))
    with outputs["sample"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_row", "wild_type_name", "mutation", "observed_ddg_kcal_mol", "predicted_ddg_kcal_mol", "residual_kcal_mol"))
        writer.writeheader()
        for index in sample_indexes:
            row = validation_rows[int(index)]
            writer.writerow({
                "source_row": row["source_row"], "wild_type_name": row["wild_type_name"], "mutation": row["mutation"],
                "observed_ddg_kcal_mol": format(validation_targets[index], ".6f"), "predicted_ddg_kcal_mol": format(validation_predictions[index], ".6f"), "residual_kcal_mol": format(validation_predictions[index] - validation_targets[index], ".6f"),
            })
    save_model(outputs["model"], selected_model)

    results = {
        "experiment_version": "phase2_ridge_baseline_v1",
        "source_records_sha256": sha256sum(args.records),
        "split_manifest": {"version": split_summary["config"]["split_version"], "seed": split_summary["config"]["seed"]},
        "selection_protocol": "Train model/scaler on train only; select alpha by validation MAE only; test split was not loaded or evaluated.",
        "test_set_accessed": False,
        "feature_schema": {"names": FEATURE_NAMES, "count": len(FEATURE_NAMES)},
        "candidate_alphas": candidates,
        "selected_alpha": selected_alpha,
        "training_mean_control_validation_metrics": regression_metrics(validation_targets, mean_predictions),
        "selected_validation_metrics": regression_metrics(validation_targets, validation_predictions),
        "counts": {"train": len(train_rows), "validation": len(validation_rows)},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "artifacts": {name: path.name for name, path in outputs.items()},
    }
    outputs["metrics"].write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
