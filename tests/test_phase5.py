"""Product boundary tests without model downloads or reopening held-out evaluation."""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARTIFACT_NAMES = ("selection.json", "verification_report.json", "test_metrics.json",
                  "test_evaluation_guard.json", "best_model.pt")
HAS_LOCAL_PHASE4 = all((ROOT / "artifacts/phase4" / name).is_file() for name in ARTIFACT_NAMES)

from mutantscope.api import create_app
from mutantscope.data_pipeline import mutate
from mutantscope.esm_features import EncoderSpec
from mutantscope.inference import (AMINO_ACID_ORDER, InferenceService, InputError,
    position_mutations, validate_mutation, validate_position, validate_sequence,
    verify_deployment_artifacts)
from mutantscope.scan_jobs import ScanJobs


class FakeEncoder:
    device = "cpu"

    def __init__(self):
        self.calls = []

    @staticmethod
    def residues(sequence):
        # Distinct residues, sites, and channels make feature swaps observable.
        return np.stack([np.arange(480, dtype=np.float32) / 100
            + AMINO_ACID_ORDER.index(residue) + position / 10
            for position, residue in enumerate(sequence)])

    def encode(self, sequences):
        self.calls.append(list(sequences))
        return [self.residues(sequence) for sequence in sequences]


class CapturingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = []

    def forward(self, features):
        self.features.append(features.clone())
        return features[:, 0] + features[:, 960] / 10


def fake_service():
    service = InferenceService.__new__(InferenceService)
    service.spec = EncoderSpec()
    service.encoder = FakeEncoder()
    service.model = CapturingHead()
    service.mean = np.arange(3840, dtype=np.float32) / 500
    service.scale = 1 + np.arange(3840, dtype=np.float32) / 1000
    service.batch_size = 7
    service.lock = threading.RLock()
    service.wild_cache = OrderedDict()
    service.prediction_cache = OrderedDict()
    service.wild_cache_bytes = 0
    service.wild_cache_budget = 32 * 1024 * 1024
    service.prediction_cache_limit = 8192
    service.info = {"checkpoint_sha256": "fixture-checkpoint", "model_version": "fixture-v1",
        "unit": "kcal/mol", "positive_means": "stabilization",
        "limitations": ["Research estimate; requires experimental validation."]}
    return service


class DeferredExecutor:
    """Keep queued/running transitions deterministic without worker timing races."""
    def __init__(self):
        self.pending = []

    def submit(self, function, *args):
        self.pending.append((function, args))

    def run_next(self):
        function, args = self.pending.pop(0)
        function(*args)

    def shutdown(self, **kwargs):
        self.pending.clear()


def deferred_jobs(service, **kwargs):
    jobs = ScanJobs(service, **kwargs)
    jobs.executor.shutdown(wait=True)
    jobs.executor = DeferredExecutor()
    return jobs


class Phase5ValidationTests(unittest.TestCase):
    def test_documented_case_and_outer_whitespace_canonicalization(self):
        self.assertEqual(validate_sequence(" \nacdef\t"), "ACDEF")
        self.assertEqual(validate_mutation("ACDEF", " c2v ").notation, "C2V")

    def test_reject_invalid_sequence_and_never_truncate(self):
        for sequence in ("", " \t", "AC DE", "AC\nDE", ">protein\nACDE", "ACBX", "ACUO", "AC*"):
            with self.subTest(sequence=sequence), self.assertRaises(InputError) as error:
                validate_sequence(sequence)
            self.assertEqual(error.exception.code, "invalid_sequence")
        self.assertEqual(len(validate_sequence("A" * 1024)), 1024)
        with self.assertRaises(InputError) as error:
            validate_sequence("A" * 1025)
        self.assertEqual(error.exception.code, "sequence_too_long")

    def test_reject_position_mismatch_multi_noop_and_noncanonical_mutation(self):
        for position in (0, 5, -1, True, 1.0, "1"):
            with self.subTest(position=position), self.assertRaises(InputError) as error:
                validate_position("ACDE", position)
            self.assertEqual(error.exception.code, "position_out_of_range")
        cases = {"C1V": "wild_type_mismatch", "A5V": "position_out_of_range",
            "A0V": "invalid_mutation", "A1A": "invalid_mutation", "A1V:C2D": "invalid_mutation",
            "A1V,C2D": "invalid_mutation", "A1B": "invalid_mutation", "A 1V": "invalid_mutation"}
        for notation, code in cases.items():
            with self.subTest(notation=notation), self.assertRaises(InputError) as error:
                validate_mutation("ACDE", notation)
            self.assertEqual(error.exception.code, code)

    def test_position_has_exactly_19_unique_non_wildtype_variants(self):
        for position in (1, 4):
            mutations = position_mutations("ACDE", position)
            self.assertEqual(len(mutations), 19)
            self.assertEqual(len({mutation.notation for mutation in mutations}), 19)
            self.assertEqual({mutation.mutant for mutation in mutations}, set(AMINO_ACID_ORDER) - {"ACDE"[position - 1]})
            self.assertTrue(all(mutation.position == position for mutation in mutations))


class Phase5InferenceTests(unittest.TestCase):
    def test_feature_layout_normalization_and_prediction_use_checkpoint_contract(self):
        service = fake_service()
        sequence = "ACDE"
        mutation = validate_mutation(sequence, "C2V")
        result = service.predict_one(sequence, mutation.notation)
        wild = FakeEncoder.residues(sequence)
        variant = FakeEncoder.residues(mutate(sequence, mutation))
        sw, sm, gw, gm = wild[1], variant[1], wild.mean(axis=0), variant.mean(axis=0)
        raw = np.concatenate((sw, sm, sm - sw, np.abs(sm - sw), gw, gm, gm - gw, np.abs(gm - gw)))
        expected = (raw - service.mean) / service.scale
        np.testing.assert_array_equal(service.model.features[0].numpy(), expected[None, :])
        self.assertEqual(result["ddg_kcal_mol"], float(np.float32(expected[0] + expected[960] / 10)))
        self.assertEqual(result["positive_means"], "stabilization")
        self.assertEqual(result["unit"], "kcal/mol")

    def test_cache_reuse_isolated_returns_and_checkpoint_keys(self):
        service = fake_service()
        first = service.predict_one("ACDE", "A1V")
        first["ddg_kcal_mol"] = 999
        second = service.predict_one(" acde ", "a1v")
        self.assertNotEqual(second["ddg_kcal_mol"], 999)
        self.assertEqual(len(service.encoder.calls), 2)
        service.predict_one("ACDE", "C2V")
        self.assertEqual(len(service.encoder.calls), 3)  # Wild type reused.
        service.info["checkpoint_sha256"] = "another-checkpoint"
        service.predict_one("ACDE", "A1V")
        self.assertEqual(len(service.encoder.calls), 4)  # Predictions cannot cross checkpoints.

    def test_caches_stay_bounded_and_batch_larger_than_cache_returns_all_rows(self):
        service = fake_service()
        service.prediction_cache_limit = 2
        service.wild_cache_budget = 4 * 480 * 4
        predictions = service.scan_position("ACDE", 1)["predictions"]
        self.assertEqual(len(predictions), 19)
        self.assertEqual(len(service.prediction_cache), 2)
        service.predict_one("VCDE", "V1A")
        self.assertEqual(len(service.wild_cache), 1)
        self.assertLessEqual(service.wild_cache_bytes, service.wild_cache_budget)
        service.wild_cache_budget = 1
        service.predict_one("ACDEF", "A1V")
        self.assertEqual(service.wild_cache_bytes, 0)
        self.assertFalse(service.wild_cache)

    def test_duplicate_mutations_keep_requested_order_and_progress_cancellation(self):
        service = fake_service()
        mutations = [validate_mutation("ACDE", name) for name in ("A1V", "C2V", "A1V")]
        rows = service.predict_mutations("ACDE", mutations)
        self.assertEqual([row["mutation"] for row in rows], [mutation.notation for mutation in mutations])
        service.batch_size = 2
        received = []
        rows = service.predict_mutations("ACDE", position_mutations("ACDE", 1),
            progress=lambda batch: received.extend(batch), cancelled=lambda: len(received) >= 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows, received)

    def test_nonfinite_predictions_fail_instead_of_returning_invalid_json(self):
        service = fake_service()
        service.model = lambda features: torch.full((len(features),), float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Nonfinite inference predictions"):
            service.predict_one("ACDE", "A1V")

    @unittest.skipUnless(HAS_LOCAL_PHASE4, "Local ignored Phase 4 deployment artifacts are unavailable")
    def test_real_deployment_artifacts_pass_without_reopening_evaluation(self):
        selection, verification, metrics = verify_deployment_artifacts(ROOT / "artifacts/phase4")
        self.assertEqual(selection["status"], "frozen")
        self.assertIs(verification["test_artifacts_verified"], True)
        self.assertEqual(metrics["status"], "evaluated")

    @unittest.skipUnless(HAS_LOCAL_PHASE4, "Local ignored Phase 4 deployment artifacts are unavailable")
    def test_artifact_gate_rejects_unverified_or_changed_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            for name in ARTIFACT_NAMES:
                shutil.copy2(ROOT / "artifacts/phase4" / name, destination / name)
            selection = json.loads((destination / "selection.json").read_text())
            selection["normalizer_fitted_split"] = "validation"
            (destination / "selection.json").write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError, "verified, frozen"):
                verify_deployment_artifacts(destination)


class Phase5JobTests(unittest.TestCase):
    def test_full_scan_exact_19_times_length_and_progress_is_visible(self):
        service = fake_service()
        jobs = deferred_jobs(service)
        self.addCleanup(jobs.close)
        snapshots = []
        original = service.predict_mutations
        def observed(sequence, mutations, *, progress, cancelled):
            def record(batch):
                progress(batch)
                snapshots.append(jobs.snapshot(submitted["job_id"]))
            return original(sequence, mutations, progress=record, cancelled=cancelled)
        service.predict_mutations = observed
        submitted = jobs.submit(" acd ")
        self.assertEqual(submitted["status"], "queued")
        self.assertEqual(submitted["total"], 57)
        jobs.executor.run_next()
        result = jobs.snapshot(submitted["job_id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed"], 57)
        self.assertEqual(len(result["predictions"]), 57)
        self.assertEqual(len({row["mutation"] for row in result["predictions"]}), 57)
        self.assertEqual({position: sum(row["position"] == position for row in result["predictions"])
            for position in (1, 2, 3)}, {1: 19, 2: 19, 3: 19})
        self.assertTrue(all(snapshot["status"] == "running" and snapshot["predictions"] == [] for snapshot in snapshots))
        self.assertEqual([snapshot["completed"] for snapshot in snapshots], list(range(7, 57, 7)) + [57])
        result["predictions"][0]["mutation"] = "corrupted"
        self.assertNotEqual(jobs.snapshot(submitted["job_id"])["predictions"][0]["mutation"], "corrupted")

    def test_cancel_queued_and_running_jobs_and_hide_partial_results(self):
        service = fake_service()
        jobs = deferred_jobs(service)
        self.addCleanup(jobs.close)
        queued = jobs.submit("AC")
        self.assertEqual(jobs.cancel(queued["job_id"])["status"], "cancelled")
        jobs.executor.run_next()
        self.assertFalse(service.encoder.calls)
        original = service.predict_mutations
        running = jobs.submit("AC")
        def cancel_after_batch(sequence, mutations, *, progress, cancelled):
            def record(batch):
                progress(batch)
                jobs.cancel(running["job_id"])
            return original(sequence, mutations, progress=record, cancelled=cancelled)
        service.predict_mutations = cancel_after_batch
        jobs.executor.run_next()
        result = jobs.snapshot(running["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["completed"], 7)
        self.assertEqual(result["predictions"], [])

    def test_capacity_eviction_expiry_and_unknown_job(self):
        jobs = deferred_jobs(fake_service(), max_pending=2, max_jobs=2, ttl_seconds=10)
        self.addCleanup(jobs.close)
        first = jobs.submit("AC")
        jobs.submit("AC")
        with self.assertRaises(InputError) as error:
            jobs.submit("AC")
        self.assertEqual(error.exception.code, "scan_capacity")
        jobs.executor.run_next()
        jobs.submit("AC")
        self.assertEqual(len(jobs.jobs), 2)
        with self.assertRaises(InputError) as error:
            jobs.snapshot(first["job_id"])
        self.assertEqual(error.exception.code, "job_not_found")
        jobs.executor.run_next()
        terminal = next(key for key, job in jobs.jobs.items() if job["status"] == "completed")
        jobs.jobs[terminal]["finished_at"] = 0
        with patch("mutantscope.scan_jobs.time.monotonic", return_value=11):
            with self.assertRaises(InputError) as error:
                jobs.snapshot(terminal)
        self.assertEqual(error.exception.code, "job_not_found")

    def test_cancelled_queued_job_eviction_does_not_break_worker(self):
        jobs = deferred_jobs(fake_service(), max_pending=2, max_jobs=1)
        self.addCleanup(jobs.close)
        cancelled = jobs.submit("AC")
        jobs.cancel(cancelled["job_id"])
        replacement = jobs.submit("AC")
        self.assertNotIn(cancelled["job_id"], jobs.jobs)
        jobs.executor.run_next()  # A cancelled queued task can outlive its evicted metadata.
        jobs.executor.run_next()
        self.assertEqual(jobs.snapshot(replacement["job_id"])["status"], "completed")

    def test_cancel_churn_cannot_exceed_submitted_work_capacity(self):
        service = fake_service()
        jobs = deferred_jobs(service, max_pending=2)
        self.addCleanup(jobs.close)
        jobs.submit("AC")
        cancelled = jobs.submit("AC")
        jobs.cancel(cancelled["job_id"])
        for _ in range(10):
            with self.assertRaises(InputError) as error:
                jobs.submit("AC")
            self.assertEqual(error.exception.code, "scan_capacity")
        self.assertEqual(jobs.outstanding, 2)
        self.assertEqual(len(jobs.executor.pending), 2)
        jobs.executor.run_next()
        calls_after_first_job = len(service.encoder.calls)
        self.assertEqual(jobs.outstanding, 1)
        jobs.executor.run_next()  # Consuming cancelled work releases its queue slot.
        self.assertEqual(jobs.outstanding, 0)
        self.assertEqual(len(service.encoder.calls), calls_after_first_job)
        self.assertEqual(jobs.submit("AC")["status"], "queued")

    def test_job_failure_does_not_expose_internal_errors_or_partial_estimates(self):
        service = fake_service()
        def broken(sequence, mutations, *, progress, cancelled):
            progress([{"internal": "partial"}])
            raise RuntimeError("secret-local-path")
        service.predict_mutations = broken
        jobs = deferred_jobs(service)
        self.addCleanup(jobs.close)
        submitted = jobs.submit("AC")
        with self.assertLogs("mutantscope.scan_jobs", level="ERROR"):
            jobs.executor.run_next()
        result = jobs.snapshot(submitted["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["predictions"], [])
        self.assertNotIn("secret-local-path", result["error"])


class Phase5ApiTests(unittest.TestCase):
    def test_ready_prediction_position_and_provenance_endpoints(self):
        service = fake_service()
        with TestClient(create_app(service=service)) as client:
            self.assertEqual(client.get("/health").json()["status"], "ready")
            self.assertEqual(client.get("/model-info").json(), service.info)
            prediction = client.post("/predict", json={"sequence": " acde ", "mutation": "c2v"})
            self.assertEqual(prediction.status_code, 200)
            self.assertEqual(prediction.json()["mutation"], "C2V")
            self.assertIsInstance(prediction.json()["ddg_kcal_mol"], float)
            position = client.post("/scan/position", json={"sequence": "ACDE", "position": 4})
            self.assertEqual(position.status_code, 200)
            self.assertEqual(len(position.json()["predictions"]), 19)
            schema = client.get("/openapi.json").json()
            self.assertIn("Prediction", schema["components"]["schemas"])
            self.assertIn("/scan/jobs/{job_id}", schema["paths"])

    def test_strict_request_types_extra_fields_and_domain_error_formats(self):
        cases = [("/predict", {"sequence": "ACDE", "mutation": "A1V", "confidence": True}, "invalid_request"),
            ("/predict", {"sequence": 123, "mutation": "A1V"}, "invalid_request"),
            ("/scan/position", {"sequence": "ACDE", "position": "1"}, "invalid_request"),
            ("/scan/position", {"sequence": "ACDE", "position": True}, "invalid_request"),
            ("/predict", {"sequence": "AC DE", "mutation": "A1V"}, "invalid_sequence"),
            ("/predict", {"sequence": "ACDE", "mutation": "C1V"}, "wild_type_mismatch"),
            ("/predict", {"sequence": "ACDE", "mutation": "A1V:C2D"}, "invalid_mutation"),
            ("/predict", {"sequence": "A" * 1025, "mutation": "A1V"}, "sequence_too_long")]
        with TestClient(create_app(service=fake_service())) as client:
            for endpoint, body, code in cases:
                with self.subTest(endpoint=endpoint, body=body):
                    response = client.post(endpoint, json=body)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["error"]["code"], code)
                    self.assertIsInstance(response.json()["error"]["message"], str)
            response = client.get("/scan/jobs/nonexistent")
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["error"]["code"], "job_not_found")

    def test_protein_submit_poll_and_delete_endpoints(self):
        with TestClient(create_app(service=fake_service())) as client:
            submitted = client.post("/scan/protein", json={"sequence": "AC"})
            self.assertEqual(submitted.status_code, 202)
            key = submitted.json()["job_id"]
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                response = client.get(f"/scan/jobs/{key}")
                if response.json()["status"] == "completed":
                    break
                time.sleep(.01)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "completed")
            self.assertEqual(len(response.json()["predictions"]), 38)
            self.assertEqual(client.delete(f"/scan/jobs/{key}").json()["status"], "completed")

    def test_scan_queue_capacity_cancel_and_missing_job_http_statuses(self):
        service = fake_service()
        app = create_app(service=service)
        with TestClient(app) as client:
            app.state.jobs.close()
            app.state.jobs = deferred_jobs(service, max_pending=1)
            submitted = client.post("/scan/protein", json={"sequence": "AC"})
            self.assertEqual(submitted.status_code, 202)
            full = client.post("/scan/protein", json={"sequence": "AC"})
            self.assertEqual(full.status_code, 429)
            self.assertEqual(full.json()["error"]["code"], "scan_capacity")
            cancelled = client.delete(f"/scan/jobs/{submitted.json()['job_id']}")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["status"], "cancelled")
            missing = client.delete("/scan/jobs/missing")
            self.assertEqual(missing.status_code, 404)
            self.assertEqual(missing.json()["error"]["code"], "job_not_found")

    def test_csv_export_requires_completion_and_preserves_all_sorted_estimates(self):
        service = fake_service()
        app = create_app(service=service)
        with TestClient(app) as client:
            app.state.jobs.close()
            app.state.jobs = deferred_jobs(service)
            submitted = client.post("/scan/protein", json={"sequence": "ACD"})
            self.assertEqual(submitted.status_code, 202)
            key = submitted.json()["job_id"]
            endpoint = f"/scan/jobs/{key}/results.csv"
            queued = client.get(endpoint)
            self.assertEqual(queued.status_code, 409)
            self.assertEqual(queued.json()["error"]["code"], "http_error")
            app.state.jobs.executor.run_next()
            download = client.get(endpoint)
            self.assertEqual(download.status_code, 200)
            self.assertTrue(download.headers["content-type"].startswith("text/csv"))
            self.assertEqual(download.headers["cache-control"], "no-store")
            self.assertEqual(download.headers["content-disposition"],
                'attachment; filename="mutantscope-protein-scan.csv"')
            reader = csv.DictReader(io.StringIO(download.text))
            fields = ["mutation", "position", "wild_type", "mutant", "ddg_kcal_mol",
                "model_version", "unit", "positive_means"]
            self.assertEqual(reader.fieldnames, fields)
            rows = list(reader)
            self.assertEqual(len(rows), 57)
            expected = sorted(app.state.jobs.snapshot(key)["predictions"],
                key=lambda row: (-row["ddg_kcal_mol"], row["mutation"]))
            self.assertEqual([row["mutation"] for row in rows], [row["mutation"] for row in expected])
            self.assertEqual({row["mutation"] for row in rows},
                {mutation.notation for position in range(1, 4) for mutation in position_mutations("ACD", position)})
            for row, original in zip(rows, expected, strict=True):
                self.assertEqual(set(row), set(fields))
                self.assertEqual(int(row["position"]), original["position"])
                self.assertEqual(float(row["ddg_kcal_mol"]), original["ddg_kcal_mol"])
                self.assertEqual(row["wild_type"], "ACD"[int(row["position"]) - 1])
                self.assertNotEqual(row["wild_type"], row["mutant"])
                self.assertEqual(row["model_version"], service.info["model_version"])
                self.assertEqual(row["unit"], "kcal/mol")
                self.assertEqual(row["positive_means"], "stabilization")
            missing = client.get("/scan/jobs/missing/results.csv")
            self.assertEqual(missing.status_code, 404)
            self.assertEqual(missing.json()["error"]["code"], "job_not_found")

    def test_unavailable_model_and_runtime_failure_return_actionable_503(self):
        def unavailable(*args, **kwargs):
            raise FileNotFoundError("secret-local-path")
        with self.assertLogs("mutantscope.api", level="ERROR"), TestClient(create_app(service_factory=unavailable), raise_server_exceptions=False) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 503)
            self.assertFalse(health.json()["model_ready"])
            for method, endpoint, body in (("get", "/model-info", None),
                ("post", "/predict", {"sequence": "AC", "mutation": "A1V"}),
                ("post", "/scan/protein", {"sequence": "AC"})):
                response = getattr(client, method)(endpoint, **({"json": body} if body else {}))
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("secret-local-path", response.text)
                self.assertIn("error", response.json())
        service = fake_service()
        service.predict_one = lambda *args: (_ for _ in ()).throw(RuntimeError("private failure"))
        with self.assertLogs("mutantscope.api", level="ERROR"), TestClient(create_app(service=service), raise_server_exceptions=False) as client:
            response = client.post("/predict", json={"sequence": "AC", "mutation": "A1V"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "inference_failed")
            self.assertNotIn("private failure", response.text)


if __name__ == "__main__":
    unittest.main()
