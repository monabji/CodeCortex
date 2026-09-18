"""Independently audit Phase 3 cache coverage, provenance and actual ESM vectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.esm_features import (BASE_BLOCKS, FEATURE_BLOCKS, EncoderSpec, FrozenEsmEncoder,
    assemble_features, file_hash, iter_cached_features, load_feature_records, load_sequence_cache,
    load_shard, sequence_cache_key, write_json_atomic)


def verify(args: argparse.Namespace) -> dict:
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = EncoderSpec(**manifest["encoder"])
    if (manifest["status"] != "complete" or spec.fingerprint != manifest["encoder_fingerprint"]
            or manifest["feature_blocks"] != list(FEATURE_BLOCKS)
            or manifest["base_blocks"] != list(BASE_BLOCKS)
            or manifest["feature_dimension"] != spec.feature_dimension):
        raise AssertionError("Incomplete/incompatible cache manifest")
    if any(manifest[field] is not False for field in
           ("labels_used", "test_features_extracted", "test_set_evaluated")):
        raise AssertionError("Phase 3 must be label-free and development-only")
    if (manifest["source_records_sha256"] != file_hash(args.records)
            or manifest["split_summary_sha256"] != file_hash(args.split_summary)):
        raise AssertionError("Source/split provenance mismatch")
    split_summary = json.loads(args.split_summary.read_text(encoding="utf-8"))
    if (manifest["split_version"] != split_summary["config"]["split_version"]
            or manifest["split_seed"] != split_summary["config"]["seed"]):
        raise AssertionError("Frozen split configuration changed")
    records = load_feature_records(args.records, tuple(manifest["splits"]))
    if manifest["limit_per_split"]:
        if not args.allow_smoke:
            raise AssertionError("Smoke cache cannot prove full Phase 3 completion")
        records = [r for split in manifest["splits"] for r in
                   [r for r in records if r.split == split][:manifest["limit_per_split"]]]
    else:
        for split in manifest["splits"]:
            if sum(r.split == split for r in records) != split_summary["splits"][split]["records"]:
                raise AssertionError("Phase 1 partition coverage mismatch")
    expected = {r.source_row: r for r in records}
    expected_wt = {sequence_cache_key(r.wild_type_sequence, spec): r.wild_type_sequence for r in records}
    if len(expected) != manifest["records_total"] or manifest["records_completed"] != len(expected):
        raise AssertionError("Incomplete/duplicate row accounting")
    if (manifest["unique_wild_types"] != len(expected_wt)
            or len(manifest["wild_type_cache"]) != len(expected_wt)):
        raise AssertionError("Wild-type cache count mismatch")
    checked_wt = set()
    for entry in manifest["wild_type_cache"]:
        if entry["key"] not in expected_wt or entry["key"] in checked_wt:
            raise AssertionError("Duplicate/unexpected wild-type cache")
        path = args.cache_dir / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise AssertionError("Wild-type cache checksum mismatch")
        load_sequence_cache(path, expected_wt[entry["key"]], spec)
        checked_wt.add(entry["key"])
    if checked_wt != set(expected_wt):
        raise AssertionError("Missing wild-type cache entries")
    sample_ids = set(np.random.default_rng(20260918).choice(sorted(expected),
        size=min(args.reencode_samples, len(expected)), replace=False).tolist())
    # Also cover mutations at the biological first and last residue if present.
    if args.reencode_samples:
        for split in manifest["splits"]:
            for predicate in (lambda r: r.position == 1, lambda r: r.position == len(r.wild_type_sequence)):
                record = next((r for r in records if r.split == split and predicate(r)), None)
                if record:
                    sample_ids.add(record.source_row)
    samples = {}
    seen: set[int] = set()
    seen_files = set()
    counts = Counter()
    for entry in manifest["shards"]:
        path = args.cache_dir / entry["file"]
        if entry["file"] in seen_files or file_hash(path) != entry["sha256"]:
            raise AssertionError("Duplicate/corrupt feature shard")
        seen_files.add(entry["file"])
        base, rows, _ = load_shard(path, spec)
        if len(rows) != entry["records"] or any(int(r) not in expected or int(r) in seen for r in rows):
            raise AssertionError("Duplicate/unexpected feature row")
        matching = [expected[int(r)] for r in rows]
        if any(r.split != entry["split"] for r in matching):
            raise AssertionError("Feature shard crosses partition boundaries")
        load_shard(path, spec, matching)
        with np.load(path, allow_pickle=False) as shard:
            if set(shard.files) != {"base", "source_rows", "keys", "fingerprint"}:
                raise AssertionError("Unexpected fields in label-free cache")
        features = assemble_features(base).reshape(len(rows), 8, spec.hidden_size)
        np.testing.assert_array_equal(features[:, 2], base[:, 1] - base[:, 0])
        np.testing.assert_array_equal(features[:, 3], np.abs(features[:, 2]))
        np.testing.assert_array_equal(features[:, 6], base[:, 3] - base[:, 2])
        np.testing.assert_array_equal(features[:, 7], np.abs(features[:, 6]))
        for i, source_row in enumerate(rows):
            if int(source_row) in sample_ids:
                samples[int(source_row)] = base[i].copy()
        seen.update(int(r) for r in rows)
        counts.update({entry["split"]: len(rows)})
    if seen != set(expected) or dict(counts) != manifest["split_counts"]:
        raise AssertionError("Feature cache does not cover exactly the selected partitions")
    # Exercise the public reader as used by the future regression-head trainer.
    for split in manifest["splits"]:
        reader_count = sum(len(rows) for rows, features in iter_cached_features(args.cache_dir, split)
                           if features.shape == (len(rows), spec.feature_dimension))
        if reader_count != counts[split]:
            raise AssertionError("Feature reader coverage/shape mismatch")
    checks = {}
    verification_runtime = {}
    if sample_ids:
        encoder = FrozenEsmEncoder(spec, args.device, args.local_files_only, manifest.get("cpu_threads", 8))
        verification_runtime = encoder.runtime
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACD", "WY"]
        batched = encoder.encode(sequences)
        single = [encoder.encode([sequence])[0] for sequence in sequences]
        padding_error = max(float(np.max(np.abs(a - b))) for a, b in zip(batched, single, strict=True))
        for a, b in zip(batched, single, strict=True):
            np.testing.assert_allclose(a, b, atol=3e-5, rtol=3e-5)
        ordered = [expected[source_row] for source_row in sorted(sample_ids)]
        fresh_wt = encoder.encode([r.wild_type_sequence for r in ordered])
        fresh_mutant = encoder.encode([r.mutant_sequence for r in ordered])
        max_error = 0.0
        for record, wild, mutant in zip(ordered, fresh_wt, fresh_mutant, strict=True):
            recomputed = np.stack((wild[record.position - 1], mutant[record.position - 1],
                                   wild.mean(axis=0), mutant.mean(axis=0)))
            np.testing.assert_allclose(samples[record.source_row], recomputed, atol=3e-5, rtol=3e-5)
            max_error = max(max_error, float(np.max(np.abs(samples[record.source_row] - recomputed))))
        if (encoder.model.training or any(p.requires_grad or p.grad is not None for p in encoder.model.parameters())):
            raise AssertionError("Encoder not frozen during verification")
        checks = {"reencoded_records": len(ordered), "max_reencoding_absolute_error": max_error,
                  "max_padding_absolute_error": padding_error, "encoder_frozen": True,
                  "first_last_residue_checks": True, "canonical_residue_vocabulary_checked": True}
    result = {"status": "verified", "full_development_cache": manifest["limit_per_split"] == 0
              and set(manifest["splits"]) == {"train", "validation"},
              "split_counts": dict(counts), "unique_wild_types": len(expected_wt),
              "shards": len(manifest["shards"]), "feature_dimension": spec.feature_dimension,
              "manifest_sha256": file_hash(manifest_path), "source_records_sha256": file_hash(args.records),
              "encoder": manifest["encoder"], "labels_used": False,
              "test_features_extracted": False, "test_set_evaluated": False,
              "live_encoder_checks": checks, "verification_runtime": verification_runtime}
    if args.report:
        write_json_atomic(args.report, result)
        if samples:
            # A few vector diagnostics for human inspection, never ddG predictions.
            examples = []
            for split in manifest["splits"]:
                for source_row in sorted(samples):
                    record = expected[source_row]
                    if record.split == split:
                        base = samples[source_row]
                        examples.append({"source_row": source_row, "split": split,
                            "mutation": record.mutation, "sequence_length": len(record.wild_type_sequence),
                            "feature_dimension": spec.feature_dimension,
                            "site_difference_l2": float(np.linalg.norm(base[1] - base[0])),
                            "global_difference_l2": float(np.linalg.norm(base[3] - base[2]))})
            path = args.report.parent / "feature_sample.csv"
            temporary = path.with_suffix(".csv.part")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
                writer.writeheader()
                writer.writerows(examples)
            temporary.replace(path)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/phase3/esm2_site_global_v1")
    parser.add_argument("--reencode-samples", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.reencode_samples < 0:
        parser.error("reencode-samples must be nonnegative")
    verify(args)


if __name__ == "__main__":
    main()
