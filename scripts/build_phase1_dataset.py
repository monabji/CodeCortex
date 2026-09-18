"""Build validated MegaScale records and a cluster-safe split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.data_pipeline import PipelineConfig, build_phase1_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/Processed_K50_dG_datasets.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--force", action="store_true", help="Replace existing local processed artifacts")
    args = parser.parse_args()
    config = PipelineConfig(seed=args.seed)
    summary = build_phase1_dataset(args.archive, args.output_dir, args.manifest_dir, config, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
