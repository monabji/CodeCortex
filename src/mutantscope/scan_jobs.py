"""Bounded, cancellable in-memory protein scans for a single local API process."""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from mutantscope.inference import InputError, position_mutations, validate_sequence


class ScanJobs:
    def __init__(self, service, *, max_jobs: int = 8, max_pending: int = 2, ttl_seconds: int = 3600):
        self.service = service
        self.max_jobs, self.max_pending, self.ttl_seconds = max_jobs, max_pending, ttl_seconds
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="protein-scan")
        self.jobs = {}
        self.outstanding = 0

    def _prune(self):
        now = time.monotonic()
        for key, job in list(self.jobs.items()):
            if job["status"] in ("completed", "failed", "cancelled") and now - job["finished_at"] > self.ttl_seconds:
                del self.jobs[key]

    def submit(self, value: str) -> dict:
        sequence = validate_sequence(value)
        with self.lock:
            self._prune()
            # Cancelled tasks still occupy the executor queue until consumed.
            # Count submitted work, not just public job states, to bound memory.
            if self.outstanding >= self.max_pending:
                raise InputError("scan_capacity", "The scan queue is full. Wait for an active job to finish or cancel it.")
            while len(self.jobs) >= self.max_jobs:
                terminal = [key for key, job in self.jobs.items() if job["status"] in ("completed", "failed", "cancelled")]
                if not terminal:
                    raise InputError("scan_capacity", "The scan queue is full. Try again later.")
                del self.jobs[terminal[0]]
            key = uuid.uuid4().hex
            self.jobs[key] = {"job_id": key, "status": "queued", "completed": 0, "total": len(sequence) * 19,
                "sequence_length": len(sequence), "predictions": [], "error": None,
                "cancel": threading.Event(), "finished_at": None}
            self.outstanding += 1
            try:
                self.executor.submit(self._run, key, sequence)
            except Exception:
                self.outstanding -= 1
                del self.jobs[key]
                raise
            return self.snapshot(key)

    def _run(self, key: str, sequence: str):
        try:
            self._work(key, sequence)
        finally:
            with self.lock:
                self.outstanding -= 1

    def _work(self, key: str, sequence: str):
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job["cancel"].is_set():
                return
            job["status"] = "running"

        def progress(batch):
            with self.lock:
                job["predictions"].extend(batch)
                job["completed"] += len(batch)

        try:
            mutations = [mutation for position in range(1, len(sequence) + 1) for mutation in position_mutations(sequence, position)]
            self.service.predict_mutations(sequence, mutations, progress=progress, cancelled=job["cancel"].is_set)
            with self.lock:
                job["status"] = "cancelled" if job["cancel"].is_set() else "completed"
                if job["status"] == "cancelled":
                    job["predictions"].clear()
                job["finished_at"] = time.monotonic()
        except Exception:
            with self.lock:
                job["status"] = "failed"
                job["error"] = "Model inference failed. Check the API logs and model/device availability, then submit a new scan."
                job["predictions"].clear()
                job["finished_at"] = time.monotonic()
            import logging
            logging.getLogger(__name__).exception("Protein scan failed")

    def snapshot(self, key: str) -> dict:
        with self.lock:
            self._prune()
            if key not in self.jobs:
                raise InputError("job_not_found", "Scan job was not found or has expired. Submit a new scan.")
            job = self.jobs[key]
            return {name: ([dict(row) for row in job[name]] if job["status"] == "completed" else [])
                    if name == "predictions" else job[name]
                    for name in ("job_id", "status", "completed", "total", "sequence_length", "predictions", "error")}

    def cancel(self, key: str) -> dict:
        with self.lock:
            self.snapshot(key)
            job = self.jobs[key]
            if job["status"] in ("queued", "running"):
                job["cancel"].set()
                if job["status"] == "queued":
                    job["status"] = "cancelled"
                    job["finished_at"] = time.monotonic()
            return self.snapshot(key)

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"].set()
        self.executor.shutdown(wait=True, cancel_futures=True)
