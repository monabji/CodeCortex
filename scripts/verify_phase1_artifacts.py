"""Independently verify Phase 1 records, exclusions, and split-manifest integrity."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.data_pipeline import canonical_sequence, mutate, parse_single_substitution  # noqa: E402


def intersections(values: dict[str, set[str]]) -> list[str]:
    names = sorted(values)
    return sorted(set().union(*(values[first] & values[second] for index, first in enumerate(names) for second in names[index + 1 :])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz")
    parser.add_argument("--exclusions", type=Path, default=ROOT / "data/processed/megascale_v2_230420_phase1_exclusions.csv.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/manifests/megascale_v2_230420_split_v1.json")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    source_rows: set[int] = set()
    duplicate_keys: set[tuple[str, str]] = set()
    split_clusters: dict[str, set[str]] = defaultdict(set)
    split_proteins: dict[str, set[str]] = defaultdict(set)
    split_sequences: dict[str, set[str]] = defaultdict(set)
    cluster_assignments: dict[str, tuple[str, str]] = {}
    split_counts: Counter[str] = Counter()

    with gzip.open(args.records, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source_row = int(row["source_row"])
            if source_row in source_rows:
                raise AssertionError(f"Repeated retained source row: {source_row}")
            source_rows.add(source_row)
            mutation = parse_single_substitution(row["mutation"])
            if mutation is None:
                raise AssertionError(f"Invalid retained mutation: {row['mutation']}")
            wild_type = canonical_sequence(row["wild_type_sequence"])
            mutant = canonical_sequence(row["mutant_sequence"])
            if wild_type is None or mutant is None or mutate(wild_type, mutation) != mutant:
                raise AssertionError(f"Invalid reconstructed sequence at source row {source_row}")
            if row["ddg_unit"] != "kcal/mol" or row["ddg_positive_means"] != "stabilizing":
                raise AssertionError(f"Unexpected ddG convention at source row {source_row}")
            key = (wild_type, mutation.notation)
            if key in duplicate_keys:
                raise AssertionError(f"Duplicate retained sequence/mutation key at source row {source_row}")
            duplicate_keys.add(key)
            split = row["split"]
            assignment = (row["split_group_id"], split)
            if row["cluster"] in cluster_assignments and cluster_assignments[row["cluster"]] != assignment:
                raise AssertionError(f"Cluster assigned inconsistently: {row['cluster']}")
            cluster_assignments[row["cluster"]] = assignment
            split_counts[split] += 1
            split_clusters[split].add(row["cluster"])
            split_proteins[split].add(row["wild_type_name"])
            split_sequences[split].add(wild_type)

    manifest_assignments: dict[str, tuple[str, str]] = {}
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            manifest_assignments[row["cluster"]] = (row["split_group_id"], row["split"])
    if manifest_assignments != cluster_assignments:
        raise AssertionError("Split manifest does not exactly match retained-record assignments")

    exclusion_count = 0
    with gzip.open(args.exclusions, "rt", newline="", encoding="utf-8") as stream:
        for _ in csv.DictReader(stream):
            exclusion_count += 1
    if len(source_rows) != summary["retained_rows"]:
        raise AssertionError("Retained-record count disagrees with summary")
    if exclusion_count != sum(summary["exclusion_counts"].values()):
        raise AssertionError("Exclusion count disagrees with summary")
    if len(source_rows) + exclusion_count != summary["source_rows"]:
        raise AssertionError("Retained plus exclusions does not equal source rows")

    integrity = {
        "cluster_intersections": intersections(split_clusters),
        "protein_intersections": intersections(split_proteins),
        "wild_type_sequence_intersections": intersections(split_sequences),
    }
    if any(integrity.values()):
        raise AssertionError(f"Leakage intersection detected: {integrity}")
    if {split: split_counts[split] for split in ("train", "validation", "test")} != {
        split: summary["splits"][split]["records"] for split in ("train", "validation", "test")
    }:
        raise AssertionError("Per-split record counts disagree with summary")

    print(json.dumps({"retained_rows": len(source_rows), "exclusion_rows": exclusion_count, "splits": dict(split_counts), "integrity": integrity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
