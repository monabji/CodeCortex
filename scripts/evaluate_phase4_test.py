"""Evaluate the frozen Phase 4 MLP and Phase 2 baseline on the held-out split once."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mutantscope.baseline import FEATURE_NAMES, RidgeModel, Standardizer, feature_matrix, regression_metrics, targets
from mutantscope.esm_features import (EncoderSpec, FeatureRecord, FrozenEsmEncoder, exclusive_cache_lock,
    file_hash, load_sequence_cache, load_shard, save_npz_atomic, sequence_cache_key, write_json_atomic)
from mutantscope.primary_model import load_checkpoint, predict, selected_rows, write_predictions


def frozen_contract(output: Path, records: Path, summary: Path, baseline: Path):
    """Validate selection before the test loader is called; block repeat experiments."""
    selection = json.loads((output / "selection.json").read_text())
    if selection["status"] != "frozen" or selection["test_set_accessed"] is not False:
        raise ValueError("A complete development-only selection must be frozen first")
    contract = {"selection_sha256": file_hash(output / "selection.json"),
        "checkpoint_sha256": file_hash(output / "best_model.pt"),
        "baseline_model_sha256": file_hash(baseline),
        "source_records_sha256": file_hash(records), "split_summary_sha256": file_hash(summary)}
    if any(selection[key] != value for key, value in contract.items() if key != "selection_sha256"):
        raise ValueError("Frozen artifact/input fingerprints changed")
    development = json.loads((output / "development_verification.json").read_text())
    if (development["status"] != "verified" or development["checkpoint_sha256"] != contract["checkpoint_sha256"]
            or not development["normalization_verified_train_only"]
            or not development["exact_development_feature_join_verified"]):
        raise ValueError("Independent development verification must pass before opening test")
    guard_path = output / "test_evaluation_guard.json"
    if guard_path.exists():
        guard = json.loads(guard_path.read_text())
        if guard["contract"] != contract:
            raise ValueError("Cannot change model or configuration after opening test")
        if guard["status"] == "evaluated":
            raise FileExistsError("Test already evaluated; inspect saved results or run the artifact verifier")
    write_json_atomic(guard_path, {"status": "in_progress", "contract": contract})
    return selection, contract


def extract_test(rows, spec, destination, contract, threads=16, batch_size=32, device="cpu"):
    """Separate label-free test cache, tied to the already-frozen selection."""
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = {"contract": contract, "encoder_fingerprint": spec.fingerprint,
        "status": "in_progress", "records_total": len(rows), "records_completed": 0,
        "shards": [], "wild_type_cache": [], "runtime": {}, "runtime_history": [], "labels_used": False}
    previous_hashes = {}
    previous_devices = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["contract"] != contract or previous["encoder_fingerprint"] != spec.fingerprint:
            raise ValueError("Incompatible held-out feature cache")
        previous_hashes = {**previous.get("known_files", {}),
            **{entry["file"]: entry["sha256"] for entry in previous["shards"] + previous["wild_type_cache"]}}
        manifest["runtime"] = previous["runtime"]
        manifest["runtime_history"] = previous.get("runtime_history", [previous["runtime"]] if previous["runtime"] else [])
        previous_devices = {entry["file"]: entry.get("device", previous["runtime"].get("device", "unknown"))
                            for entry in previous["shards"]}
    manifest["known_files"] = previous_hashes.copy()
    write_json_atomic(manifest_path, manifest)
    # Reuse the strict Phase 3 sequence validator, then retain the test partition.
    records = [replace(FeatureRecord.from_row({**row, "split": "validation"}), split="test") for row in rows]
    encoder = None

    def encode(sequences):
        nonlocal encoder
        if encoder is None:
            encoder = FrozenEsmEncoder(spec, device, True, threads)
            manifest["runtime"] = encoder.runtime
            if encoder.runtime not in manifest["runtime_history"]:
                manifest["runtime_history"].append(encoder.runtime)
        return encoder.encode(sequences)

    wild_types = sorted({record.wild_type_sequence for record in records})
    wild_vectors = {}
    missing = []
    for sequence in wild_types:
        name = f"wild_type/{sequence_cache_key(sequence, spec)}.npz"
        path = destination / name
        if path.exists() and name in previous_hashes:
            if file_hash(path) != previous_hashes[name]:
                raise ValueError("Corrupt held-out wild-type cache")
            wild_vectors[sequence] = load_sequence_cache(path, sequence, spec)
        else:
            missing.append(sequence)
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        for sequence, residues in zip(batch, encode(batch), strict=True):
            key = sequence_cache_key(sequence, spec)
            save_npz_atomic(destination / f"wild_type/{key}.npz", key=np.asarray(key),
                            fingerprint=np.asarray(spec.fingerprint), residues=residues)
            wild_vectors[sequence] = residues
    manifest["wild_type_cache"] = [{"file": f"wild_type/{sequence_cache_key(sequence, spec)}.npz",
        "sha256": file_hash(destination / f"wild_type/{sequence_cache_key(sequence, spec)}.npz")}
        for sequence in wild_types]
    manifest["known_files"].update({entry["file"]: entry["sha256"] for entry in manifest["wild_type_cache"]})
    write_json_atomic(manifest_path, manifest)
    means = {sequence: residues.mean(axis=0) for sequence, residues in wild_vectors.items()}
    started = time.perf_counter()
    for start in range(0, len(records), 4096):
        chunk = records[start:start + 4096]
        name = f"shards/{start // 4096:05d}.npz"
        path = destination / name
        if path.exists() and name in previous_hashes:
            if file_hash(path) != previous_hashes[name]:
                raise ValueError("Corrupt held-out feature shard")
            load_shard(path, spec, chunk)
            shard_device = previous_devices.get(name, "unknown")
        else:
            base = np.empty((len(chunk), 4, 480), dtype=np.float32)
            for offset in range(0, len(chunk), batch_size):
                batch = chunk[offset:offset + batch_size]
                mutants = encode([record.mutant_sequence for record in batch])
                for j, (record, residues) in enumerate(zip(batch, mutants, strict=True), offset):
                    base[j] = np.stack((wild_vectors[record.wild_type_sequence][record.position - 1],
                        residues[record.position - 1], means[record.wild_type_sequence], residues.mean(axis=0)))
            if not np.isfinite(base).all():
                raise ValueError("Nonfinite held-out vectors")
            save_npz_atomic(path, fingerprint=np.asarray(spec.fingerprint), base=base,
                source_rows=np.asarray([record.source_row for record in chunk], dtype=np.int64),
                keys=np.asarray([record.key(spec) for record in chunk]))
            shard_device = device
        checksum = file_hash(path)
        manifest["shards"].append({"file": name, "sha256": checksum, "records": len(chunk), "device": shard_device})
        manifest["known_files"][name] = checksum
        manifest["records_completed"] += len(chunk)
        write_json_atomic(manifest_path, manifest)
        print(json.dumps({"event": "test_feature_shard", "completed": manifest["records_completed"],
            "total": len(records), "device": shard_device, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    manifest.update(status="complete", unique_wild_types=len(wild_types),
                    extraction_seconds=time.perf_counter() - started)
    write_json_atomic(manifest_path, manifest)
    path = destination / "test_base.npy"
    temporary = path.with_suffix(".npy.part")
    mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                      shape=(len(records), 4, 480))
    offset = 0
    for entry in manifest["shards"]:
        shard = destination / entry["file"]
        if file_hash(shard) != entry["sha256"]:
            raise ValueError("Held-out shard checksum changed")
        values, _, _ = load_shard(shard, spec, records[offset:offset + entry["records"]])
        mapped[offset:offset + len(values)] = values
        offset += len(values)
    mapped.flush()
    del mapped
    temporary.replace(path)
    manifest["base_sha256"] = file_hash(path)
    write_json_atomic(manifest_path, manifest)
    return np.load(path, mmap_mode="r", allow_pickle=False), manifest


def grouped_metrics(rows, observed, predictions, key):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row[key], []).append(index)
    return [{key: name, **regression_metrics(observed[indexes], predictions[indexes])}
            for name, indexes in sorted(groups.items())]


def save_grouped(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def evaluate(args):
    started = time.perf_counter()
    output = args.artifacts_dir
    selection, contract = frozen_contract(output, args.records, args.split_summary, args.baseline_model)
    checkpoint, model, mean, scale = load_checkpoint(output / "best_model.pt")
    if checkpoint["provenance"] != {key: selection[key] for key in checkpoint["provenance"]}:
        raise ValueError("Checkpoint/selection contract mismatch")
    if checkpoint["smoke_rows"] or checkpoint["test_set_accessed"] or checkpoint["normalizer_fitted_split"] != "train":
        raise ValueError("Invalid selected training checkpoint")
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    summary = json.loads(args.split_summary.read_text())
    rows = selected_rows(args.records, ("test",))["test"]
    if len(rows) != summary["splits"]["test"]["records"]:
        raise ValueError("Test partition count changed")
    spec = EncoderSpec(**selection["encoder"])
    if spec != EncoderSpec():
        raise ValueError("Unapproved frozen feature contract")
    base, cache_manifest = extract_test(rows, spec, args.test_cache_dir, contract, device=args.device)
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    estimates = predict(model, base, mean, scale, checkpoint["config"]["batch_size"])
    with np.load(args.baseline_model, allow_pickle=False) as saved:
        if saved["feature_names"].tolist() != list(FEATURE_NAMES):
            raise ValueError("Baseline feature schema mismatch")
        baseline = RidgeModel(float(saved["alpha"][0]), float(saved["intercept"][0]), saved["weights"],
                             Standardizer(saved["feature_mean"], saved["feature_scale"]))
    baseline_estimates = baseline.predict(feature_matrix(rows))
    observed = targets(rows)
    if any(file_hash(path) != contract[key] for path, key in
        ((output / "selection.json", "selection_sha256"), (output / "best_model.pt", "checkpoint_sha256"),
         (args.baseline_model, "baseline_model_sha256"), (args.records, "source_records_sha256"),
         (args.split_summary, "split_summary_sha256"))):
        raise ValueError("Frozen evaluation inputs changed during run")
    artifact_hashes = {}
    grouped_summary = {}
    for name, predictions in (("primary", estimates), ("baseline", baseline_estimates)):
        path = output / f"test_{name}_predictions.csv.gz"
        write_predictions(path, rows, observed, predictions)
        artifact_hashes[path.name] = file_hash(path)
        for key, suffix in (("wild_type_name", "protein"), ("cluster", "cluster")):
            groups = grouped_metrics(rows, observed, predictions, key)
            path = output / f"test_{name}_per_{suffix}_metrics.csv"
            save_grouped(path, groups)
            artifact_hashes[path.name] = file_hash(path)
            grouped_summary[f"{name}_{suffix}"] = {"count": len(groups),
                "macro_mae": float(np.mean([group["mae"] for group in groups])),
                "min_mae": min(group["mae"] for group in groups),
                "max_mae": max(group["mae"] for group in groups)}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for column, (name, predictions) in enumerate((("Primary MLP", estimates), ("Ridge baseline", baseline_estimates))):
        axes[0, column].hexbin(observed, predictions, gridsize=50, mincnt=1, bins="log")
        limits = [min(float(observed.min()), float(predictions.min())), max(float(observed.max()), float(predictions.max()))]
        axes[0, column].plot(limits, limits, "k--", linewidth=1)
        axes[0, column].set(xlabel="Observed ddG (kcal/mol)", ylabel="Predicted ddG (kcal/mol)", title=name)
        axes[1, column].hist(predictions - observed, bins=60)
        axes[1, column].set(xlabel="Residual: predicted − observed (kcal/mol)", ylabel="Records")
    figure.savefig(output / "test_diagnostics.png", dpi=160)
    plt.close(figure)
    sample_ids = np.sort(np.random.default_rng(20260918).choice(len(rows), size=5, replace=False))
    path = output / "test_sample.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_row", "mutation", "observed_ddg", "primary_ddg", "baseline_ddg"))
        writer.writeheader()
        writer.writerows({"source_row": rows[i]["source_row"], "mutation": rows[i]["mutation"],
            "observed_ddg": float(observed[i]), "primary_ddg": float(estimates[i]),
            "baseline_ddg": float(baseline_estimates[i])} for i in sample_ids)
    report = {"status": "evaluated", "contract": contract,
        "primary_metrics": regression_metrics(observed, estimates),
        "baseline_metrics": regression_metrics(observed, baseline_estimates),
        "counts": summary["splits"]["test"], "grouped_metrics": grouped_summary,
        "test_cache_manifest_sha256": file_hash(args.test_cache_dir / "manifest.json"),
        "prediction_artifacts_sha256": artifact_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "unit": "kcal/mol", "positive_means": "stabilization", "encoder_runtime": cache_manifest["runtime"],
        "encoder_runtime_history": cache_manifest["runtime_history"],
        "feature_device_counts": {device: sum(entry["records"] for entry in cache_manifest["shards"]
                                               if entry["device"] == device)
                                  for device in sorted({entry["device"] for entry in cache_manifest["shards"]})},
        "selection_protocol": "Frozen validation-selected MLP and baseline; no tuning on test results."}
    write_json_atomic(output / "test_metrics.json", report)
    write_json_atomic(output / "test_evaluation_guard.json", {"status": "evaluated", "contract": contract,
        "test_metrics_sha256": file_hash(output / "test_metrics.json")})
    print(json.dumps({"event": "test_evaluation_complete", "primary_metrics": report["primary_metrics"],
        "baseline_metrics": report["baseline_metrics"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "artifacts/phase2/ridge_baseline_model.npz")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase4")
    parser.add_argument("--test-cache-dir", type=Path, default=ROOT / "data/cache/phase4/test_v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Frozen test encoder device")
    args = parser.parse_args()
    with exclusive_cache_lock(args.artifacts_dir):
        with exclusive_cache_lock(args.test_cache_dir):
            evaluate(args)


if __name__ == "__main__":
    main()
