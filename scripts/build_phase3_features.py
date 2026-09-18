"""Build/resume the frozen ESM-2 train/validation cache; no regression training."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.esm_features import (BASE_BLOCKS, FEATURE_BLOCKS, EncoderSpec, FrozenEsmEncoder,
    exclusive_cache_lock, file_hash, load_feature_records, load_sequence_cache, load_shard, save_npz_atomic,
    sequence_cache_key, write_json_atomic)


def build(args: argparse.Namespace) -> dict:
    build_started = time.perf_counter()
    if args.batch_size < 1 or args.shard_size < 1 or args.cpu_threads < 1 or args.limit_per_split < 0:
        raise ValueError("Batch/shard sizes must be positive and limit must be nonnegative")
    spec = EncoderSpec()
    splits = tuple(args.splits)
    records = load_feature_records(args.records, splits)
    summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    input_counts = Counter(record.split for record in records)
    for split in splits:
        if input_counts[split] != summary["splits"][split]["records"]:
            raise ValueError("Phase 1 input counts do not match the frozen split summary")
    if args.limit_per_split:
        records = [r for split in splits
                   for r in [r for r in records if r.split == split][:args.limit_per_split]]
    source_hash = file_hash(args.records)
    contract = {"encoder": asdict(spec), "encoder_fingerprint": spec.fingerprint,
                "source_records_sha256": source_hash, "split_summary_sha256": file_hash(args.split_summary),
                "split_version": summary["config"]["split_version"], "split_seed": summary["config"]["seed"],
                "splits": list(splits), "limit_per_split": args.limit_per_split,
                "batch_size": args.batch_size, "requested_device": args.device,
                "cpu_threads": args.cpu_threads,
                "shard_size": args.shard_size, "feature_blocks": list(FEATURE_BLOCKS),
                "base_blocks": list(BASE_BLOCKS), "feature_dimension": spec.feature_dimension,
                "labels_used": False, "test_features_extracted": False, "test_set_evaluated": False}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "manifest.json"
    manifest = {**contract, "status": "in_progress", "shards": [], "wild_type_cache": [],
                "runtime": {"python": sys.version, "platform": platform.platform()},
                "records_completed": 0, "records_total": len(records)}
    previous_hashes = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in contract.items()):
            raise ValueError("Refusing to reuse a cache with a different source/model/schema/split contract")
        manifest["runtime"] = previous["runtime"]
        previous_hashes = {**previous.get("known_files", {}),
                           **{entry["file"]: entry["sha256"]
                              for entry in previous["shards"] + previous["wild_type_cache"]}}
    # Preserve all known checksums across multiple interrupted resume attempts.
    manifest["known_files"] = previous_hashes.copy()
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"event": "plan", "counts": dict(Counter(r.split for r in records)),
                      "unique_wild_types": len({r.wild_type_sequence for r in records}),
                      "feature_dimension": spec.feature_dimension, "cache": str(args.cache_dir)}), flush=True)
    encoder = None

    def encode(sequences: list[str]) -> list[np.ndarray]:
        nonlocal encoder
        if encoder is None:
            encoder = FrozenEsmEncoder(spec, args.device, args.local_files_only, args.cpu_threads)
            manifest["runtime"].update(encoder.runtime)
        result = []
        for start in range(0, len(sequences), args.batch_size):
            result.extend(encoder.encode(sequences[start:start + args.batch_size]))
        return result

    # Encode each wild type once, retaining all site vectors for its mutations.
    wild_types = sorted({r.wild_type_sequence for r in records})
    wt_vectors: dict[str, np.ndarray] = {}
    missing = []
    for sequence in wild_types:
        key = sequence_cache_key(sequence, spec)
        path = args.cache_dir / "wild_type" / f"{key}.npz"
        relative = path.relative_to(args.cache_dir).as_posix()
        if path.exists() and relative in previous_hashes:
            if file_hash(path) != previous_hashes[relative]:
                raise ValueError(f"Corrupt wild-type cache; refusing resume: {relative}")
            wt_vectors[sequence] = load_sequence_cache(path, sequence, spec)
        else:
            missing.append(sequence)
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        for sequence, residues in zip(batch, encode(batch), strict=True):
            key = sequence_cache_key(sequence, spec)
            save_npz_atomic(args.cache_dir / "wild_type" / f"{key}.npz",
                            key=np.asarray(key), fingerprint=np.asarray(spec.fingerprint), residues=residues)
            wt_vectors[sequence] = residues
    wt_means = {sequence: values.mean(axis=0) for sequence, values in wt_vectors.items()}
    manifest["wild_type_cache"] = [{"key": sequence_cache_key(s, spec),
        "file": f"wild_type/{sequence_cache_key(s, spec)}.npz",
        "sha256": file_hash(args.cache_dir / "wild_type" / f"{sequence_cache_key(s, spec)}.npz")}
        for s in wild_types]
    manifest["known_files"].update({entry["file"]: entry["sha256"] for entry in manifest["wild_type_cache"]})
    write_json_atomic(manifest_path, manifest)
    start_time = time.perf_counter()
    extracted, reused = 0, 0
    for split in splits:
        partition = [r for r in records if r.split == split]
        for start in range(0, len(partition), args.shard_size):
            chunk = partition[start:start + args.shard_size]
            name = f"{split}/{start // args.shard_size:05d}.npz"
            path = args.cache_dir / name
            if path.exists() and name in previous_hashes:
                if file_hash(path) != previous_hashes[name]:
                    raise ValueError(f"Corrupt feature cache; refusing resume: {name}")
                load_shard(path, spec, chunk)
                reused += len(chunk)
            else:
                base = np.empty((len(chunk), 4, spec.hidden_size), dtype=np.float32)
                for offset in range(0, len(chunk), args.batch_size):
                    batch = chunk[offset:offset + args.batch_size]
                    mutants = encode([r.mutant_sequence for r in batch])
                    for j, (record, residues) in enumerate(zip(batch, mutants, strict=True), offset):
                        base[j, 0] = wt_vectors[record.wild_type_sequence][record.position - 1]
                        base[j, 1] = residues[record.position - 1]
                        base[j, 2] = wt_means[record.wild_type_sequence]
                        base[j, 3] = residues.mean(axis=0)
                if not np.isfinite(base).all():
                    raise ValueError("Cannot cache nonfinite representations")
                save_npz_atomic(path, fingerprint=np.asarray(spec.fingerprint), base=base,
                    source_rows=np.asarray([r.source_row for r in chunk], dtype=np.int64),
                    keys=np.asarray([r.key(spec) for r in chunk]))
                extracted += len(chunk)
            checksum = file_hash(path)
            manifest["shards"].append({"file": name, "split": split, "records": len(chunk),
                                       "sha256": checksum})
            manifest["known_files"][name] = checksum
            manifest["records_completed"] += len(chunk)
            write_json_atomic(manifest_path, manifest)
            elapsed = time.perf_counter() - start_time
            rate = extracted / elapsed if extracted else 0.0
            print(json.dumps({"event": "shard", "split": split, "completed": manifest["records_completed"],
                "total": len(records), "new_records_per_second": round(rate, 2), "reused": reused,
                "eta_seconds": round((len(records) - manifest["records_completed"]) / rate) if rate else None}), flush=True)
    # No source or split file may change while extraction is running.
    if file_hash(args.records) != source_hash or file_hash(args.split_summary) != contract["split_summary_sha256"]:
        raise ValueError("Source/split artifact changed during extraction")
    manifest.update(status="complete", split_counts=dict(Counter(r.split for r in records)),
                    unique_wild_types=len(wild_types), extraction_seconds=time.perf_counter() - start_time,
                    build_seconds=time.perf_counter() - build_started,
                    cache_bytes=sum(p.stat().st_size for p in args.cache_dir.rglob("*.npz")))
    if encoder is not None and encoder.device == "cuda":
        manifest["runtime"]["cuda_peak_memory_allocated_bytes"] = encoder.torch.cuda.max_memory_allocated()
    write_json_atomic(manifest_path, manifest)
    if args.report:
        # Only aggregate reproducibility data are committed; vector/index files stay local.
        report = {k: v for k, v in manifest.items() if k not in ("shards", "wild_type_cache", "known_files")}
        report["shard_count"] = len(manifest["shards"])
        report["manifest_sha256"] = file_hash(manifest_path)
        write_json_atomic(args.report, report)
    print(json.dumps({"event": "complete", "records": len(records), "cache_bytes": manifest["cache_bytes"]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/phase3/esm2_site_global_v1")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke check only; use a separate cache directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with exclusive_cache_lock(args.cache_dir):
        build(args)


if __name__ == "__main__":
    main()
