from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mutantscope.data_pipeline import Mutation, PipelineConfig, Record, assign_splits, deduplicate_records, parse_single_substitution, validate_source_row


def record(source_row: int, target: float) -> Record:
    return Record(source_row, f"r{source_row}", "protein", "cluster", "ACD", "VCD", Mutation("A", 1, "V"), target)


class DataPipelineTests(unittest.TestCase):
    def test_parses_only_real_single_substitutions(self) -> None:
        self.assertEqual(parse_single_substitution("a12v"), Mutation("A", 12, "V"))
        self.assertIsNone(parse_single_substitution("wt"))
        self.assertIsNone(parse_single_substitution("A12A"))
        self.assertIsNone(parse_single_substitution("insA12"))

    def test_reconstructs_wild_type_from_mutant_source_sequence(self) -> None:
        source = {"name": "sample", "aa_seq": "VCD", "mut_type": "A1V", "WT_name": "protein", "WT_cluster": "cluster", "ddG_ML": "1.25"}
        validated, reason, _ = validate_source_row(source, 2)
        self.assertIsNone(reason)
        self.assertIsNotNone(validated)
        assert validated is not None
        self.assertEqual(validated.wild_type_sequence, "ACD")
        self.assertEqual(validated.mutant_sequence, "VCD")

    def test_rejects_mutant_residue_mismatch(self) -> None:
        source = {"name": "sample", "aa_seq": "ACD", "mut_type": "A1V", "WT_name": "protein", "WT_cluster": "cluster", "ddG_ML": "1.25"}
        _, reason, _ = validate_source_row(source, 2)
        self.assertEqual(reason, "mutant_residue_mismatch")

    def test_duplicate_policy_preserves_only_exact_repeat(self) -> None:
        retained, exclusions = deduplicate_records([record(3, 1.0), record(2, 1.0), record(4, 2.0)])
        self.assertEqual(retained, [])
        self.assertEqual({reason for _, reason, _ in exclusions}, {"duplicate_conflicting_ddg"})
        retained, exclusions = deduplicate_records([record(3, 1.0), record(2, 1.0)])
        self.assertEqual([item.source_row for item in retained], [2])
        self.assertEqual(exclusions[0][1], "duplicate_exact_measurement")

    def test_group_assignment_is_deterministic_and_uses_all_splits(self) -> None:
        weights = {"a": 50, "b": 40, "c": 30, "d": 20, "e": 10}
        first = assign_splits(weights, PipelineConfig(seed=7))
        self.assertEqual(first, assign_splits(weights, PipelineConfig(seed=7)))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
