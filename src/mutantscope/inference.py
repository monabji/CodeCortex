"""Versioned Phase 5 inference using the verified, frozen Phase 4 model."""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from mutantscope.data_pipeline import Mutation, canonical_sequence, mutate, parse_single_substitution
from mutantscope.esm_features import EncoderSpec, FEATURE_BLOCKS, FrozenEsmEncoder, assemble_features, file_hash, json_hash
from mutantscope.primary_model import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY"
LIMITATIONS = [
    "Predictions are computational estimates for research and require experimental validation.",
    "The model supports single substitutions only and uses sequence information without structure or assay conditions.",
    "Evaluation used MegaScale protein-cluster-held-out data; accuracy varies across proteins and other datasets.",
    "No calibrated uncertainty or confidence interval is available.",
]


class InputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_sequence(value: str, max_residues: int = 1024) -> str:
    """Case and outer whitespace are canonicalized; internal symbols are rejected."""
    sequence = canonical_sequence(value)
    if sequence is None:
        raise InputError("invalid_sequence", "Use a non-empty sequence containing only the 20 canonical amino acids. FASTA headers and internal whitespace are unsupported.")
    if len(sequence) > max_residues:
        raise InputError("sequence_too_long", f"Sequence length must be at most {max_residues} residues; sequences are never truncated.")
    return sequence


def validate_position(sequence: str, position: int) -> int:
    if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= len(sequence):
        raise InputError("position_out_of_range", f"Position must be a whole number between 1 and {len(sequence)} (one-based indexing).")
    return position


def validate_mutation(sequence: str, value: str) -> Mutation:
    mutation = parse_single_substitution(value)
    if mutation is None:
        raise InputError("invalid_mutation", "Enter one substitution such as V42A, with different wild-type and mutant residues.")
    validate_position(sequence, mutation.position)
    if mutate(sequence, mutation) is None:
        raise InputError("wild_type_mismatch", f"Residue {mutation.position} is {sequence[mutation.position - 1]}, but the mutation specifies {mutation.wild_type}.")
    return mutation


def position_mutations(sequence: str, position: int) -> list[Mutation]:
    validate_position(sequence, position)
    wild = sequence[position - 1]
    return [Mutation(wild, position, residue) for residue in AMINO_ACID_ORDER if residue != wild]


def verify_deployment_artifacts(artifacts: Path) -> tuple[dict, dict, dict]:
    """Gate deployment without loading source records or reopening test evaluation."""
    selection = json.loads((artifacts / "selection.json").read_text(encoding="utf-8"))
    verification = json.loads((artifacts / "verification_report.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifacts / "test_metrics.json").read_text(encoding="utf-8"))
    guard = json.loads((artifacts / "test_evaluation_guard.json").read_text(encoding="utf-8"))
    digest = file_hash(artifacts / "best_model.pt")
    spec = EncoderSpec(**selection["encoder"])
    if (selection["status"] != "frozen" or spec != EncoderSpec()
            or selection["normalizer_fitted_split"] != "train"
            or selection["feature_dimension"] != 3840
            or selection["feature_blocks"] != list(FEATURE_BLOCKS)
            or selection["unit"] != "kcal/mol" or selection["positive_means"] != "stabilization"
            or selection["checkpoint_sha256"] != digest
            or verification["checkpoint_sha256"] != digest
            or verification["status"] != "verified"
            or verification["test_artifacts_verified"] is not True
            or verification["normalization_verified_train_only"] is not True
            or metrics["status"] != "evaluated"
            or metrics["contract"]["checkpoint_sha256"] != digest
            or metrics["contract"]["selection_sha256"] != file_hash(artifacts / "selection.json")
            or verification["primary_test_metrics"] != metrics["primary_metrics"]
            or guard["status"] != "evaluated"
            or guard["contract"] != metrics["contract"]
            or guard["test_metrics_sha256"] != file_hash(artifacts / "test_metrics.json")):
        raise ValueError("A verified, frozen Phase 4 experiment with matching checkpoint and metrics is required.")
    for name in ("src/mutantscope/esm_features.py", "src/mutantscope/primary_model.py"):
        recorded = {key.replace("\\", "/"): value for key, value in selection["code_sha256"].items()}
        if recorded.get(name) != file_hash(ROOT / name):
            raise ValueError("Phase 4 inference source differs from its verified implementation.")
    return selection, verification, metrics


class InferenceService:
    """One encoder per process; bounded memory caches and serialized model access."""
    def __init__(self, artifacts: Path = ROOT / "artifacts/phase4", *, device: str = "cpu",
                 batch_size: int = 32, cpu_threads: int = 8, encoder_factory=FrozenEsmEncoder):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        selection, verification, metrics = verify_deployment_artifacts(artifacts)
        payload, model, mean, scale = load_checkpoint(artifacts / "best_model.pt")
        if (payload["epoch"] != selection["selected_epoch"] or json_hash(payload["config"]) != json_hash(selection["config"])
                or payload["normalizer_fitted_split"] != "train" or payload["smoke_rows"] != 0
                or any(payload["provenance"][key] != selection[key] for key in
                       ("encoder", "feature_blocks", "feature_dimension", "split_version", "split_seed", "source_records_sha256", "unit", "positive_means"))):
            raise ValueError("Checkpoint provenance does not match the frozen selection")
        self.spec = EncoderSpec(**selection["encoder"])
        self.encoder = encoder_factory(self.spec, device=device, local_files_only=True, cpu_threads=cpu_threads)
        self.model, self.mean, self.scale = model, mean, scale
        self.model.requires_grad_(False)
        self.model.eval()
        self.batch_size = batch_size
        self.lock = threading.RLock()
        self.wild_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.prediction_cache: OrderedDict[str, dict] = OrderedDict()
        self.wild_cache_bytes = 0
        self.wild_cache_budget = 32 * 1024 * 1024
        self.prediction_cache_limit = 8192
        self.info = {
            "model_version": selection["experiment_version"], "unit": "kcal/mol",
            "positive_means": "stabilization", "checkpoint_sha256": selection["checkpoint_sha256"],
            "encoder": {key: selection["encoder"][key] for key in ("model_id", "revision", "layer")},
            "dataset": {"name": "MegaScale", "release": "v2_230420", "doi": "10.5281/zenodo.7992926"},
            "feature_schema": self.spec.schema_version,
            "split_version": selection["split_version"], "split_seed": selection["split_seed"],
            "selected_epoch": selection["selected_epoch"],
            "test_metrics": {**metrics["primary_metrics"], "proteins": metrics["counts"]["proteins"], "clusters": metrics["counts"]["clusters"]},
            "limits": {"max_residues": self.spec.max_residues},
            "limitations": list(LIMITATIONS),
            "runtime": {"device": self.encoder.device, "batch_size": batch_size},
        }

    def _wild_residues(self, sequence: str) -> np.ndarray:
        key = json_hash({"encoder": self.spec.fingerprint, "sequence": sequence})
        if key in self.wild_cache:
            self.wild_cache.move_to_end(key)
            return self.wild_cache[key]
        residues = self.encoder.encode([sequence])[0]
        while self.wild_cache and self.wild_cache_bytes + residues.nbytes > self.wild_cache_budget:
            _, removed = self.wild_cache.popitem(last=False)
            self.wild_cache_bytes -= removed.nbytes
        if residues.nbytes <= self.wild_cache_budget:
            self.wild_cache[key] = residues
            self.wild_cache_bytes += residues.nbytes
        return residues

    def _key(self, sequence: str, mutation: Mutation) -> str:
        return json_hash({"checkpoint": self.info["checkpoint_sha256"], "encoder": self.spec.fingerprint,
                          "sequence": sequence, "mutation": mutation.notation})

    def _batch(self, sequence: str, mutations: list[Mutation]) -> list[dict]:
        with self.lock:
            keys = [self._key(sequence, mutation) for mutation in mutations]
            missing = [i for i, key in enumerate(keys) if key not in self.prediction_cache]
            fresh: dict[int, dict] = {}
            if missing:
                wild = self._wild_residues(sequence)
                wild_global = wild.mean(axis=0)
                variants = [mutate(sequence, mutations[i]) for i in missing]
                encoded = self.encoder.encode(variants)
                base = np.stack([np.stack((wild[mutations[i].position - 1], variant[mutations[i].position - 1],
                            wild_global, variant.mean(axis=0))) for i, variant in zip(missing, encoded, strict=True)])
                features = (assemble_features(base) - self.mean) / self.scale
                if not np.isfinite(features).all():
                    raise RuntimeError("Nonfinite inference features")
                with torch.inference_mode():
                    estimates = self.model(torch.from_numpy(features)).cpu().numpy()
                if not np.isfinite(estimates).all():
                    raise RuntimeError("Nonfinite inference predictions")
                for i, estimate in zip(missing, estimates, strict=True):
                    mutation = mutations[i]
                    result = {"mutation": mutation.notation, "position": mutation.position,
                        "wild_type": mutation.wild_type, "mutant": mutation.mutant,
                        "ddg_kcal_mol": float(estimate), "model_version": self.info["model_version"],
                        "unit": "kcal/mol", "positive_means": "stabilization"}
                    self.prediction_cache[keys[i]] = result
                    fresh[i] = result
            # Snapshot results before eviction, including duplicate keys in a batch.
            output = [dict(fresh.get(i, self.prediction_cache.get(key))) for i, key in enumerate(keys)]
            for key in keys:
                self.prediction_cache.move_to_end(key)
            while len(self.prediction_cache) > self.prediction_cache_limit:
                self.prediction_cache.popitem(last=False)
            return output

    def predict_mutations(self, sequence: str, mutations: list[Mutation], *,
                          progress: Callable[[list[dict]], None] | None = None,
                          cancelled: Callable[[], bool] | None = None) -> list[dict]:
        sequence = validate_sequence(sequence, self.spec.max_residues)
        for mutation in mutations:
            validate_mutation(sequence, mutation.notation)
        # Attention memory grows quadratically with sequence length.
        size = min(self.batch_size, max(1, 262144 // (len(sequence) + 2) ** 2))
        output = []
        for start in range(0, len(mutations), size):
            if cancelled and cancelled():
                break
            batch = self._batch(sequence, mutations[start:start + size])
            output.extend(batch)
            if progress:
                progress(batch)
        return output

    def predict_one(self, sequence: str, mutation: str) -> dict:
        sequence = validate_sequence(sequence, self.spec.max_residues)
        return self.predict_mutations(sequence, [validate_mutation(sequence, mutation)])[0]

    def scan_position(self, sequence: str, position: int) -> dict:
        sequence = validate_sequence(sequence, self.spec.max_residues)
        predictions = self.predict_mutations(sequence, position_mutations(sequence, position))
        return {"sequence_length": len(sequence), "position": position, "predictions": predictions,
                "model_version": self.info["model_version"], "unit": "kcal/mol", "positive_means": "stabilization"}
