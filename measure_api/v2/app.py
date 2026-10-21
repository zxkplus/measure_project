"""FastAPI adapter for the replacement stateless V2 computation service."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request, Response, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from measure_api.v2.configuration import load_settings
from measure_api.v2.errors import ServiceError
from measure_api.v2.models import ErrorResponse, OperationResponse, ServiceSettings
from measure_api.v2.registry import Operation, operations
from measure_api.v2.runtime import OperationRuntime, TERMINAL_STATES

PREFIX = "/api/v2"


def create_app(settings: Optional[ServiceSettings] = None) -> FastAPI:
    """Build the headless HTTP service with explicit shared-directory settings.

    Args:
        settings: Deployment roots and resource limits shared by HTTP and compute workers.
            None loads the configured deployment settings.

    Returns:
        FastAPI application with typed routes, trace envelopes and the operation-runtime lifespan.
    """
    if settings is None:
        settings = load_settings()
    runtime = OperationRuntime(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Measurement V2 computation service", version="2.0.0",
        openapi_url=f"{PREFIX}/openapi.json", docs_url=f"{PREFIX}/docs",
        redoc_url=None, lifespan=lifespan,
        responses={
            status: {"model": ErrorResponse}
            for status in [400, 404, 409, 410, 422, 429, 500, 503]
        },
    )
    app.state.runtime = runtime
    app.state.settings = settings

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse({"error": error.payload(request.state.trace_id)}, error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        invalid_json = any(item["type"] == "json_invalid" for item in error.errors())
        detail = [
            {"field_path": ".".join(map(str, item["loc"])), "message": item["msg"],
             "type": item["type"]} for item in error.errors()
        ]
        failure = ServiceError(
            "request.invalid_json" if invalid_json else "request.validation",
            "Malformed JSON" if invalid_json else "Request validation failed",
            400 if invalid_json else 422, details={"errors": detail},
            field_path=detail[0]["field_path"] if detail else None,
        )
        return JSONResponse({"error": failure.payload(request.state.trace_id)}, failure.status)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        failure = ServiceError("request.http_error", str(error.detail), error.status_code)
        return JSONResponse({"error": failure.payload(request.state.trace_id)}, failure.status)

    @app.get(f"{PREFIX}/health/live")
    async def live() -> dict:
        return {"status": "ok"}

    @app.get(f"{PREFIX}/health/ready")
    async def ready() -> dict:
        readable = all(path.is_dir() and os.access(path, os.R_OK)
                       for path in settings.shared_roots.values())
        writable = settings.writable_roots
        aliases = settings.shared_roots if writable is None else writable
        write_ready = all(os.access(settings.shared_roots[alias], os.W_OK) for alias in aliases)
        if not runtime.ready or not readable or not write_ready:
            raise ServiceError("service.not_ready", "Workers or shared roots are unavailable", 503,
                               retryable=True)
        return {"status": "ready", "workers": settings.max_workers}

    @app.get(f"{PREFIX}/capabilities")
    async def capabilities() -> dict:
        from measure_api.v2.versions import ALGORITHM_VERSION, runtime_identity

        return {
            "api_version": "2.0", "algorithm_version": ALGORITHM_VERSION,
            "runtime_identity": runtime_identity().model_dump(mode="json"),
            "coordinate_convention": {"points": "x/y", "unit": "px", "angles": "degrees"},
            "operations": [
                {"id": item.operation_id, "path": item.path, "family": item.family,
                 "summary": item.summary, "modes": ["sync", "async"],
                 "request_schema": item.request_model.model_json_schema(),
                 "result_schema": item.result_model.model_json_schema(),
                 "output_ports": item.output_ports}
                for item in operations().values()
            ],
        }

    @app.get(f"{PREFIX}/operations/{{identity}}", response_model=OperationResponse[Any])
    async def get_operation(identity: str, wait_ms: int = Query(0, ge=0, le=60000)) -> dict:
        return await runtime.wait(identity, wait_ms)

    @app.get(f"{PREFIX}/operations")
    async def list_operations(state: Optional[str] = None, limit: int = Query(100, ge=1, le=1000)):
        return {"operations": runtime.list(state, limit)}

    @app.get(f"{PREFIX}/operations/{{identity}}/diagnostics")
    async def diagnostics(identity: str):
        return runtime.get(identity)["diagnostics"]

    @app.get(f"{PREFIX}/traces/{{trace_id}}")
    async def trace(trace_id: str):
        records = runtime.trace(trace_id)
        if not records:
            raise ServiceError("trace.not_found", "Trace does not exist", 404)
        return {"trace_id": trace_id, "operations": records}

    @app.post(f"{PREFIX}/operations/{{identity}}:cancel", response_model=OperationResponse[Any])
    async def cancel(identity: str):
        return await runtime.cancel(identity)

    for operation in operations().values():
        _add_operation(app, runtime, operation)
    return app


def _add_operation(app: FastAPI, runtime: OperationRuntime, operation: Operation) -> None:
    async def endpoint(body, request: Request, response: Response):
        result = await runtime.submit(operation.operation_id, body, request.state.trace_id,
                                      request.headers.get("X-Caller-Id", "anonymous"))
        if result["state"] not in TERMINAL_STATES:
            response.status_code = 202
            response.headers["Location"] = f"{PREFIX}/operations/{result['operation_id']}"
        elif result["state"] == "failed":
            return JSONResponse(
                {"error": result["error"], "operation_id": result["operation_id"]},
                status_code=result["diagnostics"].get("http_status", 422),
            )
        return result

    endpoint.__annotations__["body"] = operation.request_model
    app.add_api_route(
        PREFIX + operation.path, endpoint, methods=["POST"],
        response_model=OperationResponse[operation.result_model],
        operation_id=operation.operation_id, summary=operation.summary, tags=[operation.family],
        responses={202: {
            "description": "Accepted; query the retained operation",
            "model": OperationResponse[operation.result_model],
        }},
    )
