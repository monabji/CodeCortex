"""FastAPI product boundary for Phase 5; run with scripts/serve_phase5.py."""
from __future__ import annotations

import logging
import os
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from mutantscope.inference import ROOT, InferenceService, InputError
from mutantscope.scan_jobs import ScanJobs


class SequenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sequence: str = Field(min_length=1, max_length=4096)


class PredictionRequest(SequenceRequest):
    mutation: str = Field(min_length=3, max_length=32)


class PositionRequest(SequenceRequest):
    position: int = Field(ge=1, le=1024)


class Prediction(BaseModel):
    mutation: str
    position: int
    wild_type: str
    mutant: str
    ddg_kcal_mol: float
    model_version: str
    unit: Literal["kcal/mol"]
    positive_means: Literal["stabilization"]


class PositionResult(BaseModel):
    sequence_length: int
    position: int
    predictions: list[Prediction]
    model_version: str
    unit: Literal["kcal/mol"]
    positive_means: Literal["stabilization"]


class JobResult(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    completed: int
    total: int
    sequence_length: int
    predictions: list[Prediction]
    error: str | None = None


def create_app(*, service=None, service_factory=InferenceService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = service
        app.state.startup_error = None
        try:
            if app.state.service is None:
                app.state.service = service_factory(
                    Path(os.environ.get("MUTANTSCOPE_ARTIFACTS", str(ROOT / "artifacts/phase4"))),
                    device=os.environ.get("MUTANTSCOPE_DEVICE", "cpu"),
                    batch_size=int(os.environ.get("MUTANTSCOPE_BATCH_SIZE", "32")),
                    cpu_threads=int(os.environ.get("MUTANTSCOPE_CPU_THREADS", "8")))
        except Exception:
            logging.getLogger(__name__).exception("Model startup failed")
            app.state.startup_error = "Model is unavailable. Verify Phase 4 artifacts and the pinned local encoder, then restart the API."
        app.state.jobs = ScanJobs(app.state.service) if app.state.service is not None else None
        yield
        if app.state.jobs is not None:
            app.state.jobs.close()

    app = FastAPI(title="MutantScope", version="phase5_v1", lifespan=lifespan,
                  description="Single-substitution ΔΔG estimates in kcal/mol. Positive values mean stabilization. Research use; no calibrated uncertainty.")

    @app.exception_handler(InputError)
    async def input_error(request, error):
        status = 404 if error.code == "job_not_found" else 429 if error.code == "scan_capacity" else 422
        return JSONResponse(status_code=status, content={"error": {"code": error.code, "message": str(error)}})

    @app.exception_handler(RequestValidationError)
    async def schema_error(request, error):
        details = [{"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": "Check the request fields and their types.", "details": details}})

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": {"code": "http_error", "message": str(error.detail)}})

    @app.exception_handler(Exception)
    async def inference_error(request, error):
        logging.getLogger(__name__).exception("Request failed", exc_info=error)
        return JSONResponse(status_code=503, content={"error": {"code": "inference_failed", "message": "Model inference failed. Check model/device availability and try again."}})

    def ready(request: Request):
        if request.app.state.service is None:
            raise HTTPException(503, request.app.state.startup_error)
        return request.app.state.service

    @app.get("/health")
    def health(request: Request):
        available = request.app.state.service is not None
        return JSONResponse(status_code=200 if available else 503, content={"status": "ready" if available else "unavailable",
            "model_ready": available, "detail": request.app.state.startup_error})

    @app.get("/model-info")
    def model_info(request: Request):
        return ready(request).info

    @app.post("/predict", response_model=Prediction)
    def predict(body: PredictionRequest, request: Request):
        return ready(request).predict_one(body.sequence, body.mutation)

    @app.post("/scan/position", response_model=PositionResult)
    def position(body: PositionRequest, request: Request):
        return ready(request).scan_position(body.sequence, body.position)

    @app.post("/scan/protein", response_model=JobResult, status_code=202)
    def protein(body: SequenceRequest, request: Request):
        ready(request)
        return request.app.state.jobs.submit(body.sequence)

    @app.get("/scan/jobs/{job_id}", response_model=JobResult)
    def job(job_id: str, request: Request):
        ready(request)
        return request.app.state.jobs.snapshot(job_id)

    @app.delete("/scan/jobs/{job_id}", response_model=JobResult)
    def cancel(job_id: str, request: Request):
        ready(request)
        return request.app.state.jobs.cancel(job_id)

    @app.get("/scan/jobs/{job_id}/results.csv")
    def export_csv(job_id: str, request: Request):
        ready(request)
        result = request.app.state.jobs.snapshot(job_id)
        if result["status"] != "completed":
            raise HTTPException(409, "CSV results are available after successful scan completion.")
        stream = io.StringIO(newline="")
        fields = ("mutation", "position", "wild_type", "mutant", "ddg_kcal_mol", "model_version", "unit", "positive_means")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(result["predictions"], key=lambda row: (-row["ddg_kcal_mol"], row["mutation"])):
            writer.writerow(row)
        return Response(stream.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="mutantscope-protein-scan.csv"', "Cache-Control": "no-store"})

    return app


app = create_app()
