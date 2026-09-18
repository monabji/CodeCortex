"""Phase 4 regression head, bounded normalization, and provenance-safe data joins."""
from __future__ import annotations

import csv
import gzip
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mutantscope.esm_features import (EncoderSpec, FeatureRecord, assemble_features,
    file_hash, load_shard, write_json_atomic)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260918
    hidden_sizes: tuple[int, int] = (256, 128)
    dropout: float = 0.1
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    huber_delta: float = 1.0
    max_epochs: int = 60
    patience: int = 8
    min_delta: float = 0.0001
    cpu_threads: int = 8

    def __post_init__(self):
        if (min(self.hidden_sizes) < 1 or not 0 <= self.dropout < 1
                or min(self.batch_size, self.max_epochs, self.patience, self.cpu_threads) < 1
                or self.learning_rate <= 0 or self.weight_decay < 0
                or self.huber_delta <= 0 or self.min_delta < 0):
            raise ValueError("Invalid MLP training configuration")


class PrimaryMLP(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        first, second = config.hidden_sizes
        self.network = nn.Sequential(nn.Linear(3840, first), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(first, second), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(second, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def selected_rows(path: Path, splits: tuple[str, ...]) -> dict[str, list[dict]]:
    """Filter by split before interpreting sequences or targets, keeping test deferred."""
    if not splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("Invalid selected partitions")
    output = {split: [] for split in splits}
    seen = set()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in output:
                continue
            identifier = int(row["source_row"])
            label = float(row["ddg_kcal_mol"])
            if identifier in seen or not np.isfinite(label):
                raise ValueError("Duplicate source row or nonfinite target")
            seen.add(identifier)
            output[row["split"]].append(row)
    for rows in output.values():
        rows.sort(key=lambda row: int(row["source_row"]))
    return output


def prepare_development_cache(cache: Path, records: Path, split_summary: Path, destination: Path):
    """Validate every Phase 3 shard and create label-free base-vector memmaps."""
    manifest = json.loads((cache / "manifest.json").read_text())
    summary = json.loads(split_summary.read_text())
    spec = EncoderSpec(**manifest["encoder"])
    if (spec != EncoderSpec() or manifest["status"] != "complete" or manifest["limit_per_split"] != 0
            or set(manifest["splits"]) != {"train", "validation"}
            or any(manifest[field] is not False for field in
                   ("labels_used", "test_features_extracted", "test_set_evaluated"))
            or manifest["source_records_sha256"] != file_hash(records)
            or manifest["split_summary_sha256"] != file_hash(split_summary)):
        raise ValueError("Full compatible Phase 3 development cache required")
    rows = selected_rows(records, ("train", "validation"))
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    seen_files = set()
    for split, partition in rows.items():
        count = summary["splits"][split]["records"]
        if len(partition) != count or manifest["split_counts"][split] != count:
            raise ValueError("Frozen partition counts mismatch")
        index = {int(row["source_row"]): (i, row) for i, row in enumerate(partition)}
        path = destination / f"{split}_base.npy"
        temporary = path.with_suffix(".npy.part")
        mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                          shape=(count, 4, 480))
        seen = set()
        for entry in manifest["shards"]:
            if entry["split"] != split:
                continue
            shard = cache / entry["file"]
            if entry["file"] in seen_files or file_hash(shard) != entry["sha256"]:
                raise ValueError("Duplicate or corrupt Phase 3 shard")
            seen_files.add(entry["file"])
            base, identifiers, _ = load_shard(shard, spec)
            if len(identifiers) != entry["records"]:
                raise ValueError("Shard count mismatch")
            if any(int(i) not in index or int(i) in seen for i in identifiers):
                raise ValueError("Wrong partition or duplicated feature row")
            matching = [FeatureRecord.from_row(index[int(i)][1]) for i in identifiers]
            load_shard(shard, spec, matching)
            offsets = [index[int(i)][0] for i in identifiers]
            mapped[offsets] = base
            seen.update(int(i) for i in identifiers)
        if seen != set(index):
            raise ValueError("Feature coverage mismatch")
        mapped.flush()
        del mapped
        os.replace(temporary, path)
        arrays[split] = np.load(path, mmap_mode="r", allow_pickle=False)
    provenance = {"source_records_sha256": file_hash(records),
        "split_summary_sha256": file_hash(split_summary),
        "phase3_manifest_sha256": file_hash(cache / "manifest.json"),
        "encoder": asdict(spec), "feature_dimension": spec.feature_dimension,
        "feature_blocks": manifest["feature_blocks"], "split_version": manifest["split_version"],
        "split_seed": manifest["split_seed"], "unit": "kcal/mol", "positive_means": "stabilization"}
    write_json_atomic(destination / "manifest.json", {**provenance,
        "counts": {split: len(partition) for split, partition in rows.items()},
        "base_files_sha256": {split: file_hash(destination / f"{split}_base.npy") for split in rows}})
    return rows, arrays, provenance


def fit_normalizer(base: np.ndarray, chunk_size: int = 4096):
    """Parallel Welford updates in float64, using training examples only."""
    count = 0
    mean = np.zeros(3840, dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(base), chunk_size):
        features = assemble_features(np.asarray(base[start:start + chunk_size])).astype(np.float64)
        size = len(features)
        local_mean = features.mean(axis=0)
        local_m2 = np.square(features - local_mean).sum(axis=0)
        delta = local_mean - mean
        total = count + size
        m2 += local_m2 + np.square(delta) * count * size / total
        mean += delta * size / total
        count = total
    if not count:
        raise ValueError("Cannot fit normalization to empty training data")
    scale = np.sqrt(m2 / count)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def feature_batch(base, indexes, mean, scale):
    values = (assemble_features(np.asarray(base[indexes])) - mean) / scale
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite standardized features")
    return torch.from_numpy(values)


def predict(model, base, mean, scale, batch_size=512):
    model.eval()
    output = np.empty(len(base), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(base), batch_size):
            stop = min(start + batch_size, len(base))
            output[start:stop] = model(feature_batch(base, slice(start, stop), mean, scale)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite model predictions")
    return output


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainingConfig(**payload["config"])
    model = PrimaryMLP(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    mean, scale = payload["feature_mean"].numpy(), payload["feature_scale"].numpy()
    if mean.shape != (3840,) or scale.shape != (3840,) or not np.isfinite(mean).all() \
            or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid checkpoint normalization")
    return payload, model, mean, scale


def write_predictions(path: Path, rows, observed, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".gz.part")
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_row", "wild_type_name", "cluster", "mutation",
            "observed_ddg_kcal_mol", "predicted_ddg_kcal_mol", "residual_kcal_mol"))
        writer.writeheader()
        for row, label, estimate in zip(rows, observed, predictions, strict=True):
            writer.writerow({**{key: row[key] for key in ("source_row", "wild_type_name", "cluster", "mutation")},
                "observed_ddg_kcal_mol": float(label), "predicted_ddg_kcal_mol": float(estimate),
                "residual_kcal_mol": float(estimate - label)})
    os.replace(temporary, path)
