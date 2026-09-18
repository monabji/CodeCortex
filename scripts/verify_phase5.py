"""Verify real online inference, scans, provenance, and API errors (no model selection)."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient
from mutantscope.api import create_app
from mutantscope.esm_features import file_hash, write_json_atomic
from mutantscope.inference import InferenceService, AMINO_ACID_ORDER


def verify(args):
    started = time.perf_counter()
    service = InferenceService(args.artifacts_dir, device=args.device)
    # Reproduce five already-saved validation outputs; this is deployment parity,
    # not another evaluation or a hyperparameter/checkpoint selection experiment.
    with gzip.open(args.artifacts_dir / "validation_predictions.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        fixtures = list(islice(csv.DictReader(stream), 5))
    wanted = {int(row["source_row"]) for row in fixtures}
    sequences = {}
    with gzip.open(ROOT / "data/processed/megascale_v2_230420_phase1_records.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "validation" and int(row["source_row"]) in wanted:
                sequences[int(row["source_row"])] = row["wild_type_sequence"]
            if len(sequences) == len(wanted):
                break
    assert set(sequences) == wanted, "Missing validation fixtures"
    errors = []
    with TestClient(create_app(service=service)) as client:
        assert client.get("/health").json()["model_ready"] is True
        info = client.get("/model-info").json()
        assert info["checkpoint_sha256"] == service.info["checkpoint_sha256"]
        assert info["positive_means"] == "stabilization" and "uncertainty" not in info
        for fixture in fixtures:
            response = client.post("/predict", json={"sequence": sequences[int(fixture["source_row"])], "mutation": fixture["mutation"]})
            assert response.status_code == 200, response.text
            difference = abs(response.json()["ddg_kcal_mol"] - float(fixture["predicted_ddg_kcal_mol"]))
            assert difference < 2e-5, (fixture["mutation"], difference)
            errors.append(difference)
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        for position in (1, len(sequence)):
            response = client.post("/scan/position", json={"sequence": sequence, "position": position})
            assert response.status_code == 200, response.text
            predictions = response.json()["predictions"]
            assert len(predictions) == 19
            assert {row["mutant"] for row in predictions} == set(AMINO_ACID_ORDER) - {sequence[position - 1]}
            assert len({row["mutation"] for row in predictions}) == 19
            one = client.post("/predict", json={"sequence": sequence, "mutation": predictions[0]["mutation"]}).json()
            assert one == predictions[0], "Single and scan disagree"
        for body in ({"sequence": "ACX", "mutation": "A1V"}, {"sequence": "ACD", "mutation": "V1A"},
                     {"sequence": "ACD", "mutation": "A1V,C2A"}, {"sequence": "ACD", "mutation": "A1A"},
                     {"sequence": "A" * 1025, "mutation": "A1V"}):
            response = client.post("/predict", json=body)
            assert response.status_code == 422 and "error" in response.json()
        response = client.post("/scan/protein", json={"sequence": "ACD"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = client.get(f"/scan/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert job["status"] == "completed", job
        assert job["completed"] == job["total"] == 57 and len(job["predictions"]) == 57
        assert len({row["mutation"] for row in job["predictions"]}) == 57
        assert all(row["wild_type"] == "ACD"[row["position"] - 1] and row["mutant"] != row["wild_type"] for row in job["predictions"])
        exported = client.get(f"/scan/jobs/{job_id}/results.csv")
        assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        saved_rows = list(csv.DictReader(exported.text.splitlines()))
        assert len(saved_rows) == 57 and {row["mutation"] for row in saved_rows} == {row["mutation"] for row in job["predictions"]}
        assert all(row["unit"] == "kcal/mol" and row["positive_means"] == "stabilization" and row["model_version"] == service.info["model_version"] for row in saved_rows)
        assert client.get("/scan/jobs/missing").status_code == 404
    names = ("src/mutantscope/inference.py", "src/mutantscope/api.py", "src/mutantscope/scan_jobs.py", "scripts/serve_phase5.py", "scripts/verify_phase5.py", "requirements-phase5.txt")
    report = {"status": "verified", "phase": 5, "model_version": info["model_version"],
        "checkpoint_sha256": info["checkpoint_sha256"], "feature_schema": info["feature_schema"],
        "encoder": info["encoder"], "dataset": info["dataset"], "split_version": info["split_version"],
        "unit": info["unit"], "positive_means": info["positive_means"],
        "saved_validation_predictions_reproduced": len(fixtures), "max_prediction_absolute_error": max(errors),
        "first_last_position_scans_verified": True, "whole_protein_scan_count": 57,
        "api_input_errors_verified": True, "full_scan_csv_verified": True,
        "encoder_frozen": not any(p.requires_grad for p in service.encoder.model.parameters()),
        "head_frozen": not any(p.requires_grad for p in service.model.parameters()),
        "runtime": service.encoder.runtime, "code_sha256": {name: file_hash(ROOT / name) for name in names},
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Deployment compatibility check. Existing Phase 4 validation predictions reproduced; no training, tuning, new test evaluation, or Phase 6 work."}
    write_json_atomic(args.report, report)
    print(json.dumps({key: value for key, value in report.items() if key != "runtime"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase4")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/phase5/inference_verification.json")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
