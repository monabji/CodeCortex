"""Create a data-only provenance and schema audit for the MegaScale source archive.

The script intentionally uses only the Python standard library so Phase 0 can be
reproduced before the ML environment is introduced.  It never extracts or writes
variant-level data: its JSON output contains only archive metadata, CSV schemas,
and aggregate value/quality summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ARCHIVE_MD5 = "f7e8c553efee734cf161ee6f2b0a09cf"
PRIMARY_TABLE_NAME = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
TARGET_COLUMNS = ("dG_ML", "ddG_ML")
SINGLE_SUBSTITUTION = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY][0-9]+[ACDEFGHIKLMNPQRSTVWY]$")


def md5sum(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value or value == "-":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def mutation_class(value: str) -> str:
    if value == "wt":
        return "wild_type"
    if value.startswith("ins"):
        return "insertion"
    if value.startswith("del"):
        return "deletion"
    if SINGLE_SUBSTITUTION.fullmatch(value):
        return "single_substitution"
    return "other"


def audit_primary_table(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with archive.open(member, "r") as binary_stream:
        rows = csv.DictReader((line.decode("utf-8") for line in binary_stream))
        if not rows.fieldnames:
            raise ValueError(f"{member} has no header row")

        fieldnames = rows.fieldnames
        missing = Counter({field: 0 for field in fieldnames})
        non_empty = Counter({field: 0 for field in fieldnames})
        mutation_classes: Counter[str] = Counter()
        stabilizing_labels: Counter[str] = Counter()
        ddg_by_stabilizing_label: dict[str, dict[str, float | int | None]] = {}
        single_substitution_numeric_ddg_count = 0
        groups: Counter[str] = Counter()
        targets = {
            name: {
                "numeric_count": 0,
                "missing_or_filtered_count": 0,
                "empty_count": 0,
                "dash_placeholder_count": 0,
                "other_nonnumeric_count": 0,
                "minimum": None,
                "maximum": None,
            }
            for name in TARGET_COLUMNS
            if name in fieldnames
        }
        row_count = 0

        for row in rows:
            row_count += 1
            for field in fieldnames:
                value = (row.get(field) or "").strip()
                if value:
                    non_empty[field] += 1
                else:
                    missing[field] += 1

            current_mutation_class = "other"
            if "mut_type" in fieldnames:
                current_mutation_class = mutation_class((row.get("mut_type") or "").strip())
                mutation_classes[current_mutation_class] += 1
            if "Stabilizing_mut" in fieldnames:
                stabilizing_labels[(row.get("Stabilizing_mut") or "").strip()] += 1
            for group_column in ("WT_name", "WT_cluster"):
                if group_column in fieldnames:
                    groups[f"{group_column}:" + (row.get(group_column) or "").strip()] += 1

            for name, profile in targets.items():
                raw_value = (row.get(name) or "").strip()
                numeric = parse_number(raw_value)
                if numeric is None:
                    profile["missing_or_filtered_count"] += 1
                    if not raw_value:
                        profile["empty_count"] += 1
                    elif raw_value == "-":
                        profile["dash_placeholder_count"] += 1
                    else:
                        profile["other_nonnumeric_count"] += 1
                    continue
                profile["numeric_count"] += 1
                profile["minimum"] = numeric if profile["minimum"] is None else min(profile["minimum"], numeric)
                profile["maximum"] = numeric if profile["maximum"] is None else max(profile["maximum"], numeric)
                if name == "ddG_ML":
                    if current_mutation_class == "single_substitution":
                        single_substitution_numeric_ddg_count += 1
                    label = (row.get("Stabilizing_mut") or "").strip()
                    label_profile = ddg_by_stabilizing_label.setdefault(
                        label,
                        {"numeric_count": 0, "sum": 0.0, "minimum": None, "maximum": None},
                    )
                    label_profile["numeric_count"] = int(label_profile["numeric_count"]) + 1
                    label_profile["sum"] = float(label_profile["sum"]) + numeric
                    label_profile["minimum"] = (
                        numeric
                        if label_profile["minimum"] is None
                        else min(float(label_profile["minimum"]), numeric)
                    )
                    label_profile["maximum"] = (
                        numeric
                        if label_profile["maximum"] is None
                        else max(float(label_profile["maximum"]), numeric)
                    )

        for label_profile in ddg_by_stabilizing_label.values():
            label_profile["mean"] = float(label_profile.pop("sum")) / int(label_profile["numeric_count"])

    return {
        "member": member,
        "row_count": row_count,
        "columns": fieldnames,
        "non_empty_counts": dict(non_empty),
        "empty_counts": dict(missing),
        "target_profiles": targets,
        "mutation_class_counts": dict(mutation_classes),
        "stabilizing_label_counts": dict(stabilizing_labels),
        "ddg_by_stabilizing_label": ddg_by_stabilizing_label,
        "single_substitution_numeric_ddg_count": single_substitution_numeric_ddg_count,
        "unique_wild_type_protein_count": len(
            {key.removeprefix("WT_name:") for key in groups if key.startswith("WT_name:")}
        ),
        "unique_wild_type_cluster_count": len(
            {key.removeprefix("WT_cluster:") for key in groups if key.startswith("WT_cluster:")}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Checksum-verified MegaScale ZIP archive")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/audit/megascale_v2_230420.audit.json"),
        help="Aggregate-only JSON audit output",
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    if not archive_path.is_file():
        raise SystemExit(f"Archive does not exist: {archive_path}")
    archive_md5 = md5sum(archive_path)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise SystemExit(
            f"Archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {archive_md5}. "
            "Run scripts/acquire_megascale.ps1 or use the exact pinned source."
        )

    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        primary_members = [
            member.filename
            for member in members
            if Path(member.filename).name == PRIMARY_TABLE_NAME
        ]
        if len(primary_members) != 1:
            raise SystemExit(
                f"Expected exactly one {PRIMARY_TABLE_NAME} member, found {len(primary_members)}: {primary_members}"
            )
        primary_member = primary_members[0]
        result = {
            "audit_schema_version": 1,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "record": "https://zenodo.org/records/7992926",
                "doi": "10.5281/zenodo.7992926",
                "release_version": "v2_230420",
                "license": "CC-BY-4.0",
                "archive": archive_path.name,
                "archive_bytes": archive_path.stat().st_size,
                "archive_md5": archive_md5,
            },
            "archive_members": [
                {
                    "name": member.filename,
                    "compressed_bytes": member.compress_size,
                    "uncompressed_bytes": member.file_size,
                }
                for member in members
            ],
            "primary_table": audit_primary_table(archive, primary_member),
            "runtime": {"python": sys.version, "implementation": sys.implementation.name},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote aggregate audit: {args.output}")


if __name__ == "__main__":
    main()
