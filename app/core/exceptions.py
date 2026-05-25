"""Inference-service specific exceptions and error handler registration."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.common.components import C
from app.common.core.exceptions import APIException
from app.common.observability import REQUEST_ID_HEADER, get_logger, get_request_id, log_event

logger = get_logger(C.CORE_EXCEPTIONS)


class InferenceAuthError(APIException):
    def __init__(self, detail: str = "forbidden"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail, code="auth")


class InferenceConfigError(APIException):
    def __init__(self, detail: str = "config not available"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail, code="config")


class InferenceUnavailableError(APIException):
    def __init__(self, detail: str = "service not ready"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail, code="unavailable")


class InferenceTimeoutError(APIException):
    def __init__(self, detail: str = "inference timeout"):
        super().__init__(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail, code="timeout")


class ClassifyErrorResponse:
    """Error response shape for inference endpoints."""

    def __init__(self, error_code: str, message: str, request_id: str | None = None):
        self.error_code = error_code
        self.message = message
        self.request_id = request_id

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
        }


def _build_error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    request_id = get_request_id()
    body = ClassifyErrorResponse(
        error_code=error_code,
        message=message,
        request_id=request_id,
    )
    headers = {}
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=status_code,
        content=body.to_dict(),
        headers=headers,
    )


def _log_inference_error(
    request: Request, status_code: int, error_code: str, exc: Exception
) -> None:
    level = logging.ERROR if status_code >= 500 else logging.WARNING
    log_event(
        logger,
        level,
        "inference_error",
        message=f"{request.method} {request.url.path} {status_code} {error_code}",
        service="inference_service",
        request_id=get_request_id(),
        method=request.method,
        path=request.url.path,
        statusCode=status_code,
        errorCode=error_code,
        errorDetail=type(exc).__name__,
        exc_info=status_code >= 500,
    )


def install_inference_error_handlers(app: FastAPI) -> None:
    """Register inference-specific error handlers that preserve the error_code + request_id response shape."""

    @app.exception_handler(InferenceAuthError)
    async def _auth_error(request: Request, exc: InferenceAuthError) -> JSONResponse:
        _log_inference_error(request, 403, "auth", exc)
        return _build_error_response(403, "auth", exc.detail)

    @app.exception_handler(InferenceConfigError)
    async def _config_error(request: Request, exc: InferenceConfigError) -> JSONResponse:
        _log_inference_error(request, 503, "config", exc)
        return _build_error_response(503, "config", exc.detail)

    @app.exception_handler(InferenceUnavailableError)
    async def _unavailable_error(request: Request, exc: InferenceUnavailableError) -> JSONResponse:
        _log_inference_error(request, 503, "unavailable", exc)
        return _build_error_response(503, "unavailable", exc.detail)

    @app.exception_handler(InferenceTimeoutError)
    async def _timeout_error(request: Request, exc: InferenceTimeoutError) -> JSONResponse:
        _log_inference_error(request, 504, "timeout", exc)
        return _build_error_response(504, "timeout", exc.detail)

    @app.exception_handler(APIException)
    async def _api_exception_fallback(request: Request, exc: APIException) -> JSONResponse:
        error_code = getattr(exc, "code", "error")
        _log_inference_error(request, exc.status_code, error_code, exc)
        return _build_error_response(exc.status_code, error_code, exc.detail)

    @app.exception_handler(Exception)
    async def _catchall_error(request: Request, exc: Exception) -> JSONResponse:
        _log_inference_error(request, 500, "model_runtime", exc)
        return _build_error_response(500, "model_runtime", "internal inference error")
