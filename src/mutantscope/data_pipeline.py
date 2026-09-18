"""Leakage-safe Phase 1 preparation for the MegaScale single-mutation task.

The source table's ``aa_seq`` is the mutant domain sequence.  Every retained
record is therefore checked by reconstructing the wild-type sequence from the
declared mutation and recreating the mutant sequence from that reconstruction.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MUTATION_PATTERN = re.compile(r"^(?P<wild>[ACDEFGHIKLMNPQRSTVWY])(?P<position>[1-9][0-9]*)(?P<mutant>[ACDEFGHIKLMNPQRSTVWY])$")
SOURCE_MEMBER_NAME = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
SOURCE_MD5 = "f7e8c553efee734cf161ee6f2b0a09cf"
REQUIRED_COLUMNS = frozenset({"name", "aa_seq", "mut_type", "WT_name", "WT_cluster", "ddG_ML"})


@dataclass(frozen=True)
class Mutation:
    wild_type: str
    position: int
    mutant: str

    @property
    def notation(self) -> str:
        return f"{self.wild_type}{self.position}{self.mutant}"


@dataclass(frozen=True)
class Record:
    source_row: int
    source_name: str
    wild_type_name: str
    cluster: str
    wild_type_sequence: str
    mutant_sequence: str
    mutation: Mutation
    ddg_kcal_mol: float


@dataclass(frozen=True)
class PipelineConfig:
    source_release: str = "v2_230420"
    source_doi: str = "10.5281/zenodo.7992926"
    split_version: str = "megascale_cluster_split_v1"
    seed: int = 20260918
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


def parse_single_substitution(value: str) -> Mutation | None:
    match = MUTATION_PATTERN.fullmatch(value.strip().upper())
    if not match:
        return None
    mutation = Mutation(match["wild"], int(match["position"]), match["mutant"])
    return mutation if mutation.wild_type != mutation.mutant else None


def canonical_sequence(value: str) -> str | None:
    sequence = value.strip().upper()
    return sequence if sequence and set(sequence) <= AMINO_ACIDS else None


def mutate(sequence: str, mutation: Mutation) -> str | None:
    index = mutation.position - 1
    if index < 0 or index >= len(sequence) or sequence[index] != mutation.wild_type:
        return None
    return sequence[:index] + mutation.mutant + sequence[index + 1 :]


def validate_source_row(row: dict[str, str], source_row: int) -> tuple[Record | None, str | None, str]:
    mutation = parse_single_substitution(row.get("mut_type", ""))
    if mutation is None:
        return None, "mutation_not_single_substitution", row.get("mut_type", "")

    target_raw = row.get("ddG_ML", "").strip()
    try:
        target = float(target_raw)
    except ValueError:
        return None, "ddg_not_numeric", target_raw
    if not math.isfinite(target):
        return None, "ddg_not_finite", target_raw

    mutant_sequence = canonical_sequence(row.get("aa_seq", ""))
    if mutant_sequence is None:
        return None, "invalid_mutant_sequence_alphabet", row.get("aa_seq", "")
    index = mutation.position - 1
    if index >= len(mutant_sequence):
        return None, "mutation_position_out_of_range", f"position={mutation.position}; length={len(mutant_sequence)}"
    if mutant_sequence[index] != mutation.mutant:
        return None, "mutant_residue_mismatch", f"expected={mutation.mutant}; observed={mutant_sequence[index]}"

    wild_type_sequence = mutant_sequence[:index] + mutation.wild_type + mutant_sequence[index + 1 :]
    if mutate(wild_type_sequence, mutation) != mutant_sequence:
        return None, "mutant_reconstruction_failed", mutation.notation

    wild_type_name = row.get("WT_name", "").strip()
    cluster = row.get("WT_cluster", "").strip()
    if not wild_type_name:
        return None, "missing_wild_type_name", ""
    if not cluster:
        return None, "missing_cluster", ""
    return (
        Record(
            source_row=source_row,
            source_name=row.get("name", "").strip(),
            wild_type_name=wild_type_name,
            cluster=cluster,
            wild_type_sequence=wild_type_sequence,
            mutant_sequence=mutant_sequence,
            mutation=mutation,
            ddg_kcal_mol=target,
        ),
        None,
        "",
    )


def deduplicate_records(records: Iterable[Record]) -> tuple[list[Record], list[tuple[Record, str, str]]]:
    """Keep one exact repeated measurement; reject every conflicting duplicate."""
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        grouped[(record.wild_type_sequence, record.mutation.notation)].append(record)

    retained: list[Record] = []
    exclusions: list[tuple[Record, str, str]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda item: item.source_row)
        targets = [item.ddg_kcal_mol for item in group]
        if max(targets) - min(targets) > 1e-12:
            for item in group:
                exclusions.append((item, "duplicate_conflicting_ddg", f"duplicate_count={len(group)}"))
            continue
        retained.append(group[0])
        for item in group[1:]:
            exclusions.append((item, "duplicate_exact_measurement", f"kept_source_row={group[0].source_row}"))
    return retained, exclusions


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def make_split_groups(records: Iterable[Record]) -> dict[str, str]:
    """Merge source clusters if an identical reconstructed wild type joins them."""
    records = list(records)
    clusters = {record.cluster for record in records}
    union_find = UnionFind(clusters)
    sequence_clusters: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sequence_clusters[record.wild_type_sequence].add(record.cluster)
    for related_clusters in sequence_clusters.values():
        related_clusters = sorted(related_clusters)
        for cluster in related_clusters[1:]:
            union_find.union(related_clusters[0], cluster)

    components: dict[str, list[str]] = defaultdict(list)
    for cluster in sorted(clusters):
        components[union_find.find(cluster)].append(cluster)
    group_by_cluster: dict[str, str] = {}
    for component_clusters in components.values():
        stable_key = "|".join(component_clusters)
        group_id = "cluster_group_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:12]
        for cluster in component_clusters:
            group_by_cluster[cluster] = group_id
    return group_by_cluster


def assign_splits(group_weights: dict[str, int], config: PipelineConfig) -> dict[str, str]:
    fractions = {"train": config.train_fraction, "validation": config.validation_fraction, "test": config.test_fraction}
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one")
    if len(group_weights) < len(fractions):
        raise ValueError("Need at least three independent split groups")

    total = sum(group_weights.values())
    desired = {split: total * fraction for split, fraction in fractions.items()}
    assigned = {split: 0 for split in fractions}
    result: dict[str, str] = {}
    tie_breaker = random.Random(config.seed)
    ordered_groups = sorted(group_weights, key=lambda group: (-group_weights[group], group))
    for group in ordered_groups:
        weight = group_weights[group]
        scored_splits = []
        for split in fractions:
            before = (assigned[split] - desired[split]) ** 2
            after = (assigned[split] + weight - desired[split]) ** 2
            scored_splits.append((after - before, tie_breaker.random(), split))
        _, _, chosen = min(scored_splits)
        result[group] = chosen
        assigned[chosen] += weight
    if set(result.values()) != set(fractions):
        raise RuntimeError("Deterministic assignment left a split empty")
    return result


def _md5sum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_source_member(archive: zipfile.ZipFile) -> str:
    matches = [member.filename for member in archive.infolist() if Path(member.filename).name == SOURCE_MEMBER_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected one {SOURCE_MEMBER_NAME} member, found {matches}")
    return matches[0]


def _write_exclusion(writer: csv.DictWriter, source_row: int, row: dict[str, str], reason: str, detail: str) -> None:
    writer.writerow({
        "source_row": source_row,
        "source_name": row.get("name", ""),
        "wild_type_name": row.get("WT_name", ""),
        "cluster": row.get("WT_cluster", ""),
        "mutation": row.get("mut_type", ""),
        "reason": reason,
        "detail": detail,
    })


def build_phase1_dataset(source_archive: Path, output_dir: Path, manifest_dir: Path, config: PipelineConfig, force: bool = False) -> dict[str, object]:
    source_archive = source_archive.resolve()
    if _md5sum(source_archive) != SOURCE_MD5:
        raise ValueError("Source archive MD5 does not match the pinned Phase 0 release")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "megascale_v2_230420_phase1_records.csv.gz"
    exclusions_path = output_dir / "megascale_v2_230420_phase1_exclusions.csv.gz"
    manifest_path = manifest_dir / "megascale_v2_230420_split_v1.csv"
    summary_path = manifest_dir / "megascale_v2_230420_split_v1.json"
    paths = (records_path, exclusions_path, manifest_path, summary_path)
    if not force and any(path.exists() for path in paths):
        existing = ", ".join(str(path) for path in paths if path.exists())
        raise FileExistsError(f"Refusing to overwrite Phase 1 artifacts without --force: {existing}")

    exclusion_counts: Counter[str] = Counter()
    candidates: list[Record] = []
    source_rows = 0
    with zipfile.ZipFile(source_archive) as archive:
        member = _find_source_member(archive)
        with archive.open(member) as raw, gzip.open(exclusions_path, "wt", newline="", encoding="utf-8") as exclusions_stream:
            reader = csv.DictReader(line.decode("utf-8") for line in raw)
            if not reader.fieldnames or not REQUIRED_COLUMNS <= set(reader.fieldnames):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
                raise ValueError(f"Source schema missing required columns: {missing}")
            exclusion_writer = csv.DictWriter(exclusions_stream, fieldnames=("source_row", "source_name", "wild_type_name", "cluster", "mutation", "reason", "detail"))
            exclusion_writer.writeheader()
            for source_row, row in enumerate(reader, start=2):
                source_rows += 1
                record, reason, detail = validate_source_row(row, source_row)
                if record is None:
                    exclusion_counts[reason or "unknown"] += 1
                    _write_exclusion(exclusion_writer, source_row, row, reason or "unknown", detail)
                    continue
                candidates.append(record)

            protein_sequences: dict[str, set[str]] = defaultdict(set)
            protein_clusters: dict[str, set[str]] = defaultdict(set)
            for record in candidates:
                protein_sequences[record.wild_type_name].add(record.wild_type_sequence)
                protein_clusters[record.wild_type_name].add(record.cluster)
            inconsistent_proteins = {
                protein
                for protein in protein_sequences
                if len(protein_sequences[protein]) != 1 or len(protein_clusters[protein]) != 1
            }
            consistent_candidates: list[Record] = []
            for record in candidates:
                if record.wild_type_name in inconsistent_proteins:
                    exclusion_counts["inconsistent_wild_type_metadata"] += 1
                    _write_exclusion(exclusion_writer, record.source_row, {
                        "name": record.source_name, "WT_name": record.wild_type_name,
                        "WT_cluster": record.cluster, "mut_type": record.mutation.notation,
                    }, "inconsistent_wild_type_metadata", "WT_name maps to multiple reconstructed sequences or clusters")
                else:
                    consistent_candidates.append(record)

            retained, duplicate_exclusions = deduplicate_records(consistent_candidates)
            for record, reason, detail in duplicate_exclusions:
                exclusion_counts[reason] += 1
                _write_exclusion(exclusion_writer, record.source_row, {
                    "name": record.source_name, "WT_name": record.wild_type_name,
                    "WT_cluster": record.cluster, "mut_type": record.mutation.notation,
                }, reason, detail)

    group_by_cluster = make_split_groups(consistent_candidates)
    group_weights: Counter[str] = Counter(group_by_cluster[record.cluster] for record in retained)
    split_by_group = assign_splits(dict(group_weights), config)

    split_rows: Counter[str] = Counter()
    split_proteins: dict[str, set[str]] = defaultdict(set)
    split_clusters: dict[str, set[str]] = defaultdict(set)
    split_sequences: dict[str, set[str]] = defaultdict(set)
    with gzip.open(records_path, "wt", newline="", encoding="utf-8") as records_stream:
        fields = ("source_row", "source_name", "wild_type_name", "cluster", "split_group_id", "split", "wild_type_sequence", "mutant_sequence", "mutation", "position", "wild_type_residue", "mutant_residue", "ddg_kcal_mol", "ddg_unit", "ddg_positive_means")
        writer = csv.DictWriter(records_stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(retained, key=lambda item: item.source_row):
            group_id = group_by_cluster[record.cluster]
            split = split_by_group[group_id]
            writer.writerow({
                "source_row": record.source_row, "source_name": record.source_name, "wild_type_name": record.wild_type_name,
                "cluster": record.cluster, "split_group_id": group_id, "split": split,
                "wild_type_sequence": record.wild_type_sequence, "mutant_sequence": record.mutant_sequence,
                "mutation": record.mutation.notation, "position": record.mutation.position,
                "wild_type_residue": record.mutation.wild_type, "mutant_residue": record.mutation.mutant,
                "ddg_kcal_mol": format(record.ddg_kcal_mol, ".17g"), "ddg_unit": "kcal/mol",
                "ddg_positive_means": "stabilizing",
            })
            split_rows[split] += 1
            split_proteins[split].add(record.wild_type_name)
            split_clusters[split].add(record.cluster)
            split_sequences[split].add(hashlib.sha256(record.wild_type_sequence.encode("utf-8")).hexdigest())

    group_clusters: dict[str, list[str]] = defaultdict(list)
    for cluster, group_id in group_by_cluster.items():
        group_clusters[group_id].append(cluster)
    cluster_rows: Counter[str] = Counter(record.cluster for record in retained)
    cluster_proteins: dict[str, set[str]] = defaultdict(set)
    for record in retained:
        cluster_proteins[record.cluster].add(record.wild_type_name)
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_stream:
        writer = csv.DictWriter(manifest_stream, fieldnames=("split_group_id", "split", "cluster", "retained_records", "retained_proteins"))
        writer.writeheader()
        for group_id in sorted(group_clusters):
            for cluster in sorted(group_clusters[group_id]):
                writer.writerow({
                    "split_group_id": group_id, "split": split_by_group[group_id], "cluster": cluster,
                    "retained_records": cluster_rows[cluster], "retained_proteins": len(cluster_proteins[cluster]),
                })

    def intersections(values: dict[str, set[str]]) -> list[str]:
        names = sorted(values)
        return sorted(set().union(*(values[first] & values[second] for index, first in enumerate(names) for second in names[index + 1 :])))

    integrity = {
        "cluster_intersections": intersections(split_clusters),
        "protein_intersections": intersections(split_proteins),
        "wild_type_sequence_sha256_intersections": intersections(split_sequences),
    }
    if any(integrity.values()):
        raise RuntimeError(f"Leakage intersection detected: {integrity}")
    summary: dict[str, object] = {
        "pipeline_version": 1,
        "config": asdict(config),
        "source": {"archive": source_archive.name, "archive_md5": SOURCE_MD5, "member": SOURCE_MEMBER_NAME},
        "source_rows": source_rows,
        "initial_validated_candidates": len(candidates),
        "inconsistent_wild_type_proteins": len(inconsistent_proteins),
        "retained_rows": len(retained),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "splits": {
            split: {"records": split_rows[split], "proteins": len(split_proteins[split]), "clusters": len(split_clusters[split])}
            for split in ("train", "validation", "test")
        },
        "split_groups": len(group_weights),
        "integrity": integrity,
        "runtime": {"python": sys.version, "implementation": sys.implementation.name},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
