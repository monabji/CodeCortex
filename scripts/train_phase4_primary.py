"""Train the frozen-feature MLP; validation selects the checkpoint, test stays deferred."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mutantscope.baseline import regression_metrics, targets
from mutantscope.esm_features import active_package_versions, exclusive_cache_lock, file_hash, write_json_atomic
from mutantscope.primary_model import (PrimaryMLP, TrainingConfig, feature_batch, fit_normalizer,
    load_checkpoint, predict, prepare_development_cache, save_checkpoint, write_predictions)


def train(args):
    started = time.perf_counter()
    output = args.artifacts_dir
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best_model.pt", "selection.json", "test_evaluation_guard.json")):
        raise FileExistsError("Existing Phase 4 experiment: use a new artifacts directory")
    config = TrainingConfig(max_epochs=args.max_epochs, cpu_threads=args.cpu_threads,
                            batch_size=args.batch_size, patience=args.patience)
    code_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in
        (Path(__file__).resolve(), ROOT / "src/mutantscope/primary_model.py",
         ROOT / "src/mutantscope/esm_features.py", ROOT / "src/mutantscope/baseline.py")}
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    print(json.dumps({"event": "preparing_cache", "config": asdict(config)}), flush=True)
    rows, base, provenance = prepare_development_cache(args.cache_dir, args.records,
                                                      args.split_summary, args.prepared_dir)
    provenance["code_sha256"] = code_hashes
    provenance["loss"] = "HuberLoss"
    provenance["optimizer"] = "AdamW"
    if args.smoke_rows:
        rows = {split: partition[:args.smoke_rows] for split, partition in rows.items()}
        base = {split: values[:args.smoke_rows] for split, values in base.items()}
    labels = {split: targets(partition) for split, partition in rows.items()}
    mean, scale = fit_normalizer(base["train"])
    model = PrimaryMLP(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.HuberLoss(delta=config.huber_delta)
    best_mae, stale, history = float("inf"), 0, []
    patience_reference = float("inf")
    write_json_atomic(output / "training_config.json", {"config": asdict(config), **provenance,
        "smoke_rows": args.smoke_rows, "test_set_accessed": False,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "environment": {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "device": "cpu", "packages": active_package_versions()}})
    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        order = np.random.default_rng(config.seed + epoch).permutation(len(base["train"]))
        model.train()
        loss_total = 0.0
        for start in range(0, len(order), config.batch_size):
            indexes = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(feature_batch(base["train"], indexes, mean, scale))
            batch_labels = torch.from_numpy(labels["train"][indexes].astype(np.float32))
            loss = loss_fn(predictions, batch_labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indexes)
        validation = predict(model, base["validation"], mean, scale, config.batch_size)
        metrics = regression_metrics(labels["validation"], validation)
        mae = float(metrics["mae"])
        improved = mae < best_mae
        if improved:
            best_mae = mae
            save_checkpoint(output / "best_model.pt", {"model_state": model.state_dict(),
                "feature_mean": torch.from_numpy(mean), "feature_scale": torch.from_numpy(scale),
                "config": asdict(config), "provenance": provenance,
                "epoch": epoch, "validation_metrics": metrics,
                "normalizer_fitted_split": "train", "normalizer_count": len(base["train"]),
                "test_set_accessed": False, "smoke_rows": args.smoke_rows})
        if mae < patience_reference - config.min_delta:
            patience_reference, stale = mae, 0
        else:
            stale += 1
        entry = {"epoch": epoch, "train_huber_loss": loss_total / len(order),
                 "validation_metrics": metrics, "selected": improved,
                 "seconds": time.perf_counter() - epoch_started, "stale_epochs": stale}
        history.append(entry)
        write_json_atomic(output / "training_history.json", {"epochs": history,
            "test_set_accessed": False, "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps({"event": "epoch", **entry}), flush=True)
        if stale >= config.patience:
            break
    checkpoint, selected, mean, scale = load_checkpoint(output / "best_model.pt")
    predictions = predict(selected, base["validation"], mean, scale, config.batch_size)
    write_predictions(output / "validation_predictions.csv.gz", rows["validation"], labels["validation"], predictions)
    selection = {"status": "frozen" if not args.smoke_rows else "smoke_only",
        "experiment_version": "phase4_frozen_esm_mlp_v1", "config": asdict(config), **provenance,
        "checkpoint_sha256": file_hash(output / "best_model.pt"),
        "baseline_model_sha256": file_hash(ROOT / "artifacts/phase2/ridge_baseline_model.npz"),
        "validation_predictions_sha256": file_hash(output / "validation_predictions.csv.gz"),
        "selected_epoch": checkpoint["epoch"], "validation_metrics": regression_metrics(labels["validation"], predictions),
        "normalizer_fitted_split": "train", "counts": {split: len(partition) for split, partition in rows.items()},
        "selection_metric": "validation_mae", "test_set_accessed": False,
        "stop_reason": "patience" if stale >= config.patience else "max_epochs",
        "elapsed_seconds": time.perf_counter() - started}
    write_json_atomic(output / "selection.json", selection)
    print(json.dumps({"event": "training_complete", **selection}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/phase3/esm2_site_global_v1")
    parser.add_argument("--prepared-dir", type=Path, default=ROOT / "data/cache/phase4/development_v1")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase4")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_rows < 0:
        parser.error("smoke-rows must be nonnegative")
    if args.smoke_rows and args.artifacts_dir == ROOT / "artifacts/phase4":
        parser.error("Use a separate artifacts directory for smoke training")
    with exclusive_cache_lock(args.artifacts_dir):
        with exclusive_cache_lock(args.prepared_dir):
            train(args)


if __name__ == "__main__":
    main()
