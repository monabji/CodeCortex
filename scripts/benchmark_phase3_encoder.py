"""Measure label-free frozen-ESM CPU throughput before a full extraction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mutantscope.esm_features import FrozenEsmEncoder, load_feature_records, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--batches", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()
    if any(value < 1 for value in args.threads + args.batches):
        parser.error("thread/batch values must be positive")
    records = load_feature_records(ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz", ("train",))
    # Use distributed source rows so this is not a benchmark of just one length.
    sequences = [r.mutant_sequence for r in records[::max(1, len(records) // 256)][:256]]
    encoder = FrozenEsmEncoder(device="cpu", local_files_only=True)
    results = []
    for threads in args.threads:
        encoder.torch.set_num_threads(threads)
        for batch_size in args.batches:
            batch = [sequences[i % len(sequences)] for i in range(batch_size)]
            encoder.encode(batch)
            started = time.perf_counter()
            for _ in range(2):
                encoder.encode(batch)
            elapsed = time.perf_counter() - started
            result = {"threads": threads, "batch_size": batch_size,
                      "records_per_second": 2 * batch_size / elapsed}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {"runtime": encoder.runtime, "candidates": results,
              "selected": max(results, key=lambda result: result["records_per_second"]),
              "labels_used": False, "test_set_evaluated": False}
    if args.report:
        write_json_atomic(args.report, report)
    print(json.dumps({"selected": report["selected"]}), flush=True)


if __name__ == "__main__":
    main()
