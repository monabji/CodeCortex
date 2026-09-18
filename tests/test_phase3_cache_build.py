"""Exercise resume and corruption decisions without downloading an encoder."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
module_spec = importlib.util.spec_from_file_location("phase3_builder", ROOT / "scripts/build_phase3_features.py")
builder = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(builder)

from mutantscope.esm_features import exclusive_cache_lock, iter_cached_features, save_npz_atomic


class FakeEncoder:
    calls = 0

    def __init__(self, *args):
        self.device = "cpu"
        self.runtime = {"device": "synthetic_test_only"}

    def encode(self, sequences):
        FakeEncoder.calls += len(sequences)
        return [np.repeat(np.array([ord(r) / 100 for r in s], dtype=np.float32)[:, None], 480, axis=1)
                for s in sequences]


class CacheBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        records = self.directory / "records.csv.gz"
        with gzip.open(records, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("source_row", "split", "wild_type_sequence",
                "mutant_sequence", "position", "mutation", "ddg_kcal_mol"))
            writer.writeheader()
            writer.writerows([
                dict(source_row=1, split="train", wild_type_sequence="ACD", mutant_sequence="VCD",
                     position=1, mutation="A1V", ddg_kcal_mol="not_a_number_and_never_read"),
                dict(source_row=2, split="validation", wild_type_sequence="ACD", mutant_sequence="ACV",
                     position=3, mutation="D3V", ddg_kcal_mol="not_a_number_and_never_read"),
                dict(source_row=3, split="test", wild_type_sequence="invalid_and_never_encoded",
                     mutant_sequence="invalid", position=999, mutation="invalid", ddg_kcal_mol="invalid")])
        summary = self.directory / "splits.json"
        summary.write_text(json.dumps({"config": {"split_version": "fixture_v1", "seed": 1},
            "splits": {"train": {"records": 1}, "validation": {"records": 1}}}))
        self.args = argparse.Namespace(batch_size=2, shard_size=1, cpu_threads=8, limit_per_split=0,
            records=records, split_summary=summary, splits=["train", "validation"],
            cache_dir=self.directory / "cache", device="cpu", local_files_only=True, report=None)

    def tearDown(self):
        self.temporary.cleanup()

    def run_build(self):
        with patch.object(builder, "FrozenEsmEncoder", FakeEncoder), contextlib.redirect_stdout(io.StringIO()):
            return builder.build(self.args)

    def test_resume_uses_cached_vectors_and_reader_checks_provenance(self):
        FakeEncoder.calls = 0
        manifest = self.run_build()
        self.assertEqual(FakeEncoder.calls, 3)
        FakeEncoder.calls = 0
        self.run_build()
        self.assertEqual(FakeEncoder.calls, 0)
        batches = list(iter_cached_features(self.args.cache_dir, "train",
            expected_source_records_sha256=manifest["source_records_sha256"]))
        self.assertEqual(batches[0][1].shape, (1, 3840))
        with self.assertRaises(ValueError):
            list(iter_cached_features(self.args.cache_dir, "train", expected_source_records_sha256="wrong"))

    def test_resume_rejects_finite_corruption_using_prior_checksum(self):
        self.run_build()
        path = self.args.cache_dir / "train/00000.npz"
        with np.load(path) as shard:
            values = {key: shard[key] for key in shard.files}
        values["base"][0, 0, 0] += 0.5
        save_npz_atomic(path, **values)
        with self.assertRaisesRegex(ValueError, "Corrupt feature cache"):
            self.run_build()

    def test_unrecorded_orphan_shards_are_regenerated(self):
        self.run_build()
        path = self.args.cache_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["shards"] = []
        manifest["known_files"].pop("train/00000.npz")
        path.write_text(json.dumps(manifest))
        FakeEncoder.calls = 0
        self.run_build()
        self.assertEqual(FakeEncoder.calls, 1)

    def test_reader_rejects_duplicate_shard_coverage(self):
        self.run_build()
        path = self.args.cache_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["shards"].append(manifest["shards"][0])
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            list(iter_cached_features(self.args.cache_dir, "train"))

    def test_concurrent_builder_lock_is_rejected(self):
        with exclusive_cache_lock(self.args.cache_dir):
            with self.assertRaises(RuntimeError):
                with exclusive_cache_lock(self.args.cache_dir):
                    self.fail("Concurrent lock unexpectedly succeeded")


if __name__ == "__main__":
    unittest.main()
