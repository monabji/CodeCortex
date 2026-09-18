"""Frozen ESM-2 representations and a versioned, label-free mutation cache.

Only four base vectors are stored. Signed/absolute differences are assembled
on read, preserving the specified 3,840-feature layout without doubling disk.
Torch/Transformers are imported only when an actual encoder is requested.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from mutantscope.data_pipeline import canonical_sequence, mutate, parse_single_substitution

MODEL_ID = "facebook/esm2_t12_35M_UR50D"
MODEL_REVISION = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
FEATURE_BLOCKS = ("s_w", "s_m", "s_m-s_w", "abs(s_m-s_w)",
                  "g_w", "g_m", "g_m-g_w", "abs(g_m-g_w)")
BASE_BLOCKS = ("s_w", "s_m", "g_w", "g_m")


def active_package_versions() -> dict[str, str]:
    """Resolve versions in import-path order, including inherited environments.

    Distribution enumeration can include both local and shadowed global copies;
    using a last-wins comprehension reports the wrong active version.
    """
    names = {d.metadata["Name"] for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return {name: importlib.metadata.version(name) for name in sorted(names)}


@contextmanager
def exclusive_cache_lock(cache_dir: Path):
    """OS-held lock releases even after a crash, preventing concurrent builders."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / ".builder.lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another builder holds this cache; use its process or a different cache directory") from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    layer: int = 12
    hidden_size: int = 480
    schema_version: str = "esm2_site_global_difference_v1"
    inference_dtype: str = "float32"
    storage_dtype: str = "float32"
    tokenizer: str = "EsmTokenizer"
    add_special_tokens: bool = True
    padding_side: str = "right"
    truncation: bool = False
    pooling: str = "mean_residues_only"
    max_residues: int = 1024

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Encoder revision must be an immutable 40-character commit SHA")
        if not 1 <= self.layer <= 12 or self.hidden_size != 480:
            raise ValueError("This feature schema requires a 480-wide ESM-2 layer from 1 to 12")
        if self.inference_dtype != "float32" or self.storage_dtype != "float32":
            raise ValueError("This cache version supports float32 computation/storage only")
        if (self.model_id != MODEL_ID or not self.add_special_tokens or self.truncation
                or self.padding_side != "right" or self.pooling != "mean_residues_only"
                or self.tokenizer != "EsmTokenizer" or self.max_residues != 1024):
            raise ValueError("Unsupported encoder/tokenization settings")

    @property
    def fingerprint(self) -> str:
        return json_hash(asdict(self))

    @property
    def feature_dimension(self) -> int:
        return self.hidden_size * len(FEATURE_BLOCKS)


@dataclass(frozen=True)
class FeatureRecord:
    source_row: int
    split: str
    wild_type_sequence: str
    mutant_sequence: str
    position: int
    mutation: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "FeatureRecord":
        wild, mutant = row["wild_type_sequence"], row["mutant_sequence"]
        mutation = parse_single_substitution(row["mutation"])
        if (canonical_sequence(wild) != wild or canonical_sequence(mutant) != mutant
                or mutation is None or mutate(wild, mutation) != mutant
                or mutation.position != int(row["position"])):
            raise ValueError(f"Invalid Phase 1 mutation/sequence alignment: {row['source_row']}")
        if row["split"] not in ("train", "validation"):
            raise ValueError("Phase 3 extraction accepts development partitions only")
        return cls(int(row["source_row"]), row["split"], wild, mutant,
                   mutation.position, mutation.notation)

    def key(self, spec: EncoderSpec) -> str:
        return json_hash({"encoder": asdict(spec), "wild_type_sequence": self.wild_type_sequence,
                          "mutant_sequence": self.mutant_sequence, "position": self.position})


def load_feature_records(path: Path, splits: tuple[str, ...]) -> list[FeatureRecord]:
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "validation"}:
        raise ValueError("Select train and/or validation; test extraction is deferred to Phase 4")
    result = []
    seen: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            # Targets are never converted, copied, used for choices, or cached.
            if row["split"] not in splits:
                continue
            record = FeatureRecord.from_row(row)
            if record.source_row in seen:
                raise ValueError("Duplicate source row in Phase 1 input")
            seen.add(record.source_row)
            result.append(record)
    return sorted(result, key=lambda record: record.source_row)


def residue_token_indices(sequence: str, input_ids: np.ndarray, attention_mask: np.ndarray,
                          special_mask: np.ndarray, amino_ids: dict[str, int],
                          cls_id: int, eos_id: int, pad_id: int) -> np.ndarray:
    """Prove one residue per token and return biological-position -> token map."""
    if canonical_sequence(sequence) != sequence:
        raise ValueError("ESM input must be a canonical uppercase amino-acid sequence")
    if input_ids.ndim != 1 or input_ids.shape != attention_mask.shape or input_ids.shape != special_mask.shape:
        raise ValueError("Invalid token/mask shapes")
    active = np.flatnonzero(attention_mask)
    if not np.array_equal(active, np.arange(len(sequence) + 2)):
        raise ValueError("Expected right-padded CLS + residues + EOS, without truncation")
    if input_ids[0] != cls_id or input_ids[len(sequence) + 1] != eos_id:
        raise ValueError("Missing/incorrect ESM special tokens")
    indexes = np.flatnonzero((attention_mask != 0) & (special_mask == 0))
    if not np.array_equal(indexes, np.arange(1, len(sequence) + 1)):
        raise ValueError("Residue token alignment failed")
    if not np.array_equal(input_ids[indexes], [amino_ids[residue] for residue in sequence]):
        raise ValueError("Token IDs do not match the biological residues")
    if np.any(input_ids[len(sequence) + 2:] != pad_id):
        raise ValueError("Incorrect padding tokens")
    return indexes


def assemble_features(base: np.ndarray) -> np.ndarray:
    """Base order s_w,s_m,g_w,g_m -> exact eight-block ML specification."""
    if base.ndim != 3 or base.shape[1:] != (4, 480) or not np.isfinite(base).all():
        raise ValueError("Expected finite base vectors shaped (N,4,480)")
    sw, sm, gw, gm = (base[:, i, :] for i in range(4))
    ds, dg = sm - sw, gm - gw
    return np.concatenate((sw, sm, ds, np.abs(ds), gw, gm, dg, np.abs(dg)), axis=1)


class FrozenEsmEncoder:
    def __init__(self, spec: EncoderSpec = EncoderSpec(), device: str = "auto",
                 local_files_only: bool = False, cpu_threads: int = 8) -> None:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import EsmModel, EsmTokenizer

        self.torch, self.spec = torch, spec
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install the documented CUDA PyTorch wheel")
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be auto, cpu, or cuda")
        self.device = device
        if cpu_threads < 1:
            raise ValueError("CPU thread count must be positive")
        torch.set_num_threads(cpu_threads)
        torch.manual_seed(20260918)
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        model_dir = Path(__file__).resolve().parents[2] / "data/cache/phase3/encoder" / spec.revision
        # local_dir avoids Windows symlink-privilege/cache races. Pin the remote
        # revision, then fingerprint the local model files for every reuse.
        files = ("config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")
        provenance_path = model_dir / "encoder_provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance["model_id"] != spec.model_id or provenance["revision"] != spec.revision:
                raise ValueError("Incompatible local encoder snapshot")
            if any(not (model_dir / name).is_file() or file_hash(model_dir / name) != provenance["files"][name] for name in files):
                raise ValueError("Local encoder snapshot checksum mismatch")
        else:
            if local_files_only and not all((model_dir / name).is_file() for name in files):
                raise FileNotFoundError("Pinned local encoder snapshot is unavailable")
            snapshot_download(spec.model_id, revision=spec.revision, local_dir=model_dir,
                              allow_patterns=list(files), max_workers=1, local_files_only=local_files_only)
            provenance = {"model_id": spec.model_id, "revision": spec.revision,
                          "files": {name: file_hash(model_dir / name) for name in files}}
            write_json_atomic(provenance_path, provenance)
        self.tokenizer = EsmTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer.padding_side = "right"
        self.model = EsmModel.from_pretrained(model_dir, add_pooling_layer=False,
                                             use_safetensors=True, local_files_only=True)
        if (self.model.config.hidden_size != spec.hidden_size
                or self.model.config.num_hidden_layers != 12):
            raise ValueError("Downloaded model configuration differs from the pinned schema/revision")
        self.model.requires_grad_(False)
        self.model.to(device=device, dtype=torch.float32)
        self.model.eval()
        self.amino_ids = {residue: self.tokenizer.convert_tokens_to_ids(residue)
                          for residue in "ACDEFGHIKLMNPQRSTVWY"}
        if len(set(self.amino_ids.values())) != 20 or self.tokenizer.unk_token_id in self.amino_ids.values():
            raise ValueError("Canonical residue vocabulary is invalid")
        self.runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__, "device": device,
                        "cpu_threads": torch.get_num_threads(),
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                        "encoder_parameters": sum(p.numel() for p in self.model.parameters()),
                        "trainable_encoder_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                        "encoder_files_sha256": provenance["files"],
                        "packages": active_package_versions()}

    def encode(self, sequences: list[str]) -> list[np.ndarray]:
        if not sequences:
            return []
        if any(canonical_sequence(s) != s or len(s) > self.spec.max_residues for s in sequences):
            raise ValueError("Invalid/overlength sequence; silent truncation is forbidden")
        tokens = self.tokenizer(sequences, padding=True, truncation=False,
                                return_special_tokens_mask=True, return_tensors="pt")
        ids = tokens["input_ids"].numpy()
        mask = tokens["attention_mask"].numpy()
        special = tokens["special_tokens_mask"].numpy()
        mappings = [residue_token_indices(s, ids[i], mask[i], special[i], self.amino_ids,
                    self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)
                    for i, s in enumerate(sequences)]
        with self.torch.inference_mode():
            output = self.model(input_ids=tokens["input_ids"].to(self.device),
                                attention_mask=tokens["attention_mask"].to(self.device),
                                output_hidden_states=self.spec.layer != 12)
            hidden = output.last_hidden_state if self.spec.layer == 12 else output.hidden_states[self.spec.layer]
            values = hidden.float().cpu().numpy()
        residues = [values[i, indexes].copy() for i, indexes in enumerate(mappings)]
        if any(r.shape != (len(s), self.spec.hidden_size) or not np.isfinite(r).all()
               for s, r in zip(sequences, residues, strict=True)):
            raise ValueError("Invalid/nonfinite encoder representations")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("ESM encoder must remain frozen and in evaluation mode")
        return residues


def sequence_cache_key(sequence: str, spec: EncoderSpec) -> str:
    return json_hash({"encoder": asdict(spec), "sequence": sequence})


def load_sequence_cache(path: Path, sequence: str, spec: EncoderSpec) -> np.ndarray:
    with np.load(path, allow_pickle=False) as entry:
        if (str(entry["key"].item()) != sequence_cache_key(sequence, spec)
                or str(entry["fingerprint"].item()) != spec.fingerprint):
            raise ValueError(f"Stale/incompatible sequence cache: {path}")
        residues = entry["residues"]
        if residues.shape != (len(sequence), spec.hidden_size) or residues.dtype != np.float32 or not np.isfinite(residues).all():
            raise ValueError(f"Invalid sequence cache vectors: {path}")
        return residues


def load_shard(path: Path, spec: EncoderSpec,
               records: list[FeatureRecord] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as shard:
        if str(shard["fingerprint"].item()) != spec.fingerprint:
            raise ValueError(f"Stale/incompatible feature cache: {path}")
        base, source_rows, keys = shard["base"], shard["source_rows"], shard["keys"]
        if (base.shape != (len(source_rows), 4, spec.hidden_size) or base.dtype != np.float32
                or keys.shape != source_rows.shape or not np.isfinite(base).all()
                or source_rows.dtype != np.int64 or len(set(source_rows.tolist())) != len(source_rows)):
            raise ValueError(f"Invalid feature shard: {path}")
        if records is not None:
            if (not np.array_equal(source_rows, [r.source_row for r in records])
                    or not np.array_equal(keys, [r.key(spec) for r in records])):
                raise ValueError(f"Feature cache does not match current sequence/mutation inputs: {path}")
        return base, source_rows, keys


def iter_cached_features(cache_dir: Path, split: str, *, expected_source_records_sha256: str | None = None,
                         expected_split_summary_sha256: str | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Read bounded feature batches; source rows join to labels only in Phase 4."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or split not in manifest["splits"]:
        raise ValueError("Cache incomplete or requested partition unavailable")
    spec = EncoderSpec(**manifest["encoder"])
    if (manifest["feature_blocks"] != list(FEATURE_BLOCKS)
            or manifest["feature_dimension"] != spec.feature_dimension
            or manifest["encoder_fingerprint"] != spec.fingerprint
            or set(manifest["splits"]) - {"train", "validation"}
            or manifest["labels_used"] is not False):
        raise ValueError("Incompatible feature layout")
    if (expected_source_records_sha256 is not None
            and expected_source_records_sha256 != manifest["source_records_sha256"]):
        raise ValueError("Cache source provenance mismatch")
    if (expected_split_summary_sha256 is not None
            and expected_split_summary_sha256 != manifest["split_summary_sha256"]):
        raise ValueError("Cache split provenance mismatch")
    selected = [e for e in manifest["shards"] if e["split"] == split]
    if (len({e["file"] for e in selected}) != len(selected)
            or sum(e["records"] for e in selected) != manifest["split_counts"][split]):
        raise ValueError("Duplicate shards or incomplete partition coverage")
    seen: set[int] = set()
    for entry in selected:
        path = cache_dir / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Corrupt feature shard: {path}")
        base, source_rows, _ = load_shard(path, spec)
        if len(source_rows) != entry["records"] or any(int(row) in seen for row in source_rows):
            raise ValueError("Duplicate rows or incomplete feature shard")
        seen.update(int(row) for row in source_rows)
        yield source_rows, assemble_features(base)
    if len(seen) != manifest["split_counts"][split]:
        raise ValueError("Incomplete feature partition")
