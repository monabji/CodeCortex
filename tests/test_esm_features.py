from __future__ import annotations

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mutantscope.esm_features import (EncoderSpec, FeatureRecord, active_package_versions, assemble_features,
    iter_cached_features, load_sequence_cache, load_shard, residue_token_indices,
    save_npz_atomic, sequence_cache_key)


class EsmFeatureTests(unittest.TestCase):
    def test_package_versions_resolve_active_not_shadowed_distribution(self) -> None:
        distributions = [SimpleNamespace(metadata={"Name": "fsspec"}, version="2026.7.0"),
                         SimpleNamespace(metadata={"Name": "fsspec"}, version="2026.6.0")]
        with patch("mutantscope.esm_features.importlib.metadata.distributions", return_value=distributions), \
             patch("mutantscope.esm_features.importlib.metadata.version", return_value="2026.7.0") as version:
            self.assertEqual(active_package_versions(), {"fsspec": "2026.7.0"})
            version.assert_called_once_with("fsspec")

    def test_token_mapping_checks_first_last_residues_and_excludes_padding(self) -> None:
        mapping = residue_token_indices("ACD", np.array([0, 5, 23, 13, 2, 1, 1]),
            np.array([1, 1, 1, 1, 1, 0, 0]), np.array([1, 0, 0, 0, 1, 1, 1]),
            {"A": 5, "C": 23, "D": 13}, 0, 2, 1)
        self.assertEqual(mapping.tolist(), [1, 2, 3])
        with self.assertRaises(ValueError):
            residue_token_indices("ACD", np.array([0, 23, 5, 13, 2]), np.ones(5),
                np.array([1, 0, 0, 0, 1]), {"A": 5, "C": 23, "D": 13}, 0, 2, 1)

    def test_feature_layout_preserves_direction_and_magnitude(self) -> None:
        base = np.empty((2, 4, 480), dtype=np.float32)
        base[:, 0] = 3
        base[:, 1] = 1
        base[:, 2] = -2
        base[:, 3] = 4
        features = assemble_features(base).reshape(2, 8, 480)
        np.testing.assert_array_equal(features[0, :, 0], [3, 1, -2, 2, -2, 4, 6, 6])
        self.assertEqual(features.dtype, np.float32)

    def test_cache_keys_change_with_revision_layer_sequence_and_position(self) -> None:
        spec = EncoderSpec()
        record = FeatureRecord(1, "train", "ACD", "VCD", 1, "A1V")
        self.assertNotEqual(record.key(spec), record.key(replace(spec, layer=11)))
        self.assertNotEqual(record.key(spec), record.key(replace(spec, revision="a" * 40)))
        self.assertNotEqual(record.key(spec), replace(record, mutant_sequence="ACV", position=3).key(spec))

    def test_shards_reject_stale_inputs_and_nonfinite_vectors(self) -> None:
        spec = EncoderSpec()
        record = FeatureRecord(1, "train", "ACD", "VCD", 1, "A1V")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.npz"
            arrays = dict(fingerprint=np.asarray(spec.fingerprint), base=np.zeros((1, 4, 480), np.float32),
                          source_rows=np.array([1], dtype=np.int64), keys=np.array([record.key(spec)]))
            save_npz_atomic(path, **arrays)
            load_shard(path, spec, [record])
            with self.assertRaises(ValueError):
                load_shard(path, replace(spec, layer=11), [record])
            with self.assertRaises(ValueError):
                load_shard(path, spec, [replace(record, mutant_sequence="MCD")])
            arrays["base"][0, 0, 0] = np.nan
            save_npz_atomic(path, **arrays)
            with self.assertRaises(ValueError):
                load_shard(path, spec)

    def test_wild_type_cache_rejects_wrong_sequence(self) -> None:
        spec = EncoderSpec()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wt.npz"
            save_npz_atomic(path, fingerprint=np.asarray(spec.fingerprint),
                key=np.asarray(sequence_cache_key("ACD", spec)), residues=np.zeros((3, 480), np.float32))
            load_sequence_cache(path, "ACD", spec)
            with self.assertRaises(ValueError):
                load_sequence_cache(path, "VCD", spec)

    def test_records_reject_wrong_position_and_test_partition(self) -> None:
        row = {"source_row": "1", "split": "train", "wild_type_sequence": "ACD",
               "mutant_sequence": "VCD", "position": "1", "mutation": "A1V"}
        FeatureRecord.from_row(row)
        with self.assertRaises(ValueError):
            FeatureRecord.from_row({**row, "position": "3"})
        with self.assertRaises(ValueError):
            FeatureRecord.from_row({**row, "split": "test"})

    def test_reader_refuses_incomplete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"status": "in_progress"}))
            with self.assertRaises(ValueError):
                list(iter_cached_features(path, "train"))


if __name__ == "__main__":
    unittest.main()
