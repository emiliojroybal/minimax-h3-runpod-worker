from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import runpod
from pydantic import ValidationError

from h3_worker import __version__
from h3_worker.adapters import DiffusersH3Adapter, MockAdapter
from h3_worker.config import settings
from h3_worker.media import materialize_references
from h3_worker.schemas import ErrorBody, GenerationInput, GenerationResult
from h3_worker.storage import store_output


def capabilities() -> dict[str, Any]:
    return {
        "worker_version": __version__,
        "model": settings.model_id,
        "inference_mode": settings.inference_mode,
        "model_loaded": bool(DiffusersH3Adapter._pipelines),
        "modes": ["t2va", "fl2va", "ref2va"],
        "durations": {"min": 5, "max": 15},
        "short_edges": [768],
        "fps": 24,
        "max_references": {"images": 9, "videos": 3, "audio": 3, "total": 12},
        "features": {"native_audio": True, "first_frame": True, "last_frame": True, "lora": False, "open_weights_2k": False},
    }


def _progress(job: dict[str, Any], percent: int, stage: str) -> None:
    try:
        runpod.serverless.progress_update(job, {"progress": percent, "stage": stage})
    except Exception:
        # Local contract server has no RunPod job context.
        pass


def handler(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    raw_input = job.get("input", {})
    try:
        request = GenerationInput.model_validate(raw_input)
        if request.operation in {"health", "capabilities"}:
            return {"status": "ready", "capabilities": capabilities()}

        _progress(job, 5, "validated")
        adapter = MockAdapter() if settings.inference_mode == "mock" else DiffusersH3Adapter()
        with tempfile.TemporaryDirectory(prefix="h3-job-") as temp:
            workdir = Path(temp)
            local_references = materialize_references(request.references, workdir) if request.references else []
            _progress(job, 18, "references_ready")
            output_path = workdir / "output.mp4"
            _progress(job, 25, "loading_pipeline" if settings.inference_mode != "mock" else "mock_render")
            duration = adapter.generate(request, local_references, output_path)
            _progress(job, 92, "uploading_output")
            stored = store_output(output_path, request.client_job_id)
        result = GenerationResult(
            client_job_id=request.client_job_id,
            status="completed",
            mode=request.mode,
            seed=request.seed,
            output_url=stored.output_url,
            output_path=stored.output_path,
            storage=stored.storage,
            duration_seconds=duration,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        _progress(job, 100, "completed")
        return result.model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        return GenerationResult(
            client_job_id=str(raw_input.get("client_job_id", "unknown")),
            status="failed", mode=str(raw_input.get("mode", "unknown")), seed=int(raw_input.get("seed", 0) or 0),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=ErrorBody(code="VALIDATION_ERROR", message="The generation request is invalid.", details=exc.errors(include_url=False)),
        ).model_dump(mode="json", exclude_none=True)
    except Exception as exc:
        traceback.print_exc()
        return GenerationResult(
            client_job_id=str(raw_input.get("client_job_id", "unknown")),
            status="failed", mode=str(raw_input.get("mode", "unknown")), seed=int(raw_input.get("seed", 0) or 0),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=ErrorBody(code="GENERATION_FAILED", message=str(exc)),
        ).model_dump(mode="json", exclude_none=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
