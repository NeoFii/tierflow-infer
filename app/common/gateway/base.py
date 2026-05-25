"""Base gateway with shared internal HTTP call infrastructure."""

from __future__ import annotations

from typing import Any, NoReturn

from app.common.core.exceptions import ServiceUnavailableException
from app.common.internal import (
    InternalServiceError,
    InternalServiceResponseError,
    get_internal_json,
    post_internal_json,
    request_internal_json,
)


class BaseGateway:
    """Base gateway providing shared _get/_post/_request with error mapping.

    Subclasses declare base_url, timeout, and an error_map to eliminate
    repetitive _common_kwargs / _handle_error / try-except boilerplate.
    """

    def __init__(
        self,
        service_name: str,
        *,
        base_url: str,
        timeout: float = 5.0,
        error_map: dict[int, type[Exception]] | None = None,
    ) -> None:
        self.service_name = service_name
        self._base_url = base_url
        self._timeout = timeout
        self._error_map = error_map or {}

    def _common_kwargs(self) -> dict[str, Any]:
        from app.core.config import get_settings

        settings = get_settings()
        return {
            "base_url": self._base_url,
            "target_service": self.service_name,
            "secret": settings.INTERNAL_SECRET,
            "caller_service": settings.SERVICE_NAME,
            "timeout": self._timeout,
            "max_retries": settings.INTERNAL_HTTP_MAX_RETRIES,
            "retry_backoff_seconds": settings.INTERNAL_HTTP_RETRY_BACKOFF_SECONDS,
            "circuit_breaker_threshold": settings.INTERNAL_HTTP_CIRCUIT_BREAKER_THRESHOLD,
            "circuit_breaker_cooldown_seconds": (
                settings.INTERNAL_HTTP_CIRCUIT_BREAKER_COOLDOWN_SECONDS
            ),
        }

    def _handle_error(self, exc: InternalServiceError) -> NoReturn:
        if isinstance(exc, InternalServiceResponseError):
            for code, exc_class in self._error_map.items():
                if exc.status_code == code:
                    raise exc_class(detail=exc.detail or "Error") from exc
        raise ServiceUnavailableException(
            f"{self.service_name} unavailable",
        ) from exc

    async def _get(
        self,
        path: str,
        *,
        query_params: dict | list[tuple[str, object]] | None = None,
        **kwargs: Any,
    ) -> dict | None:
        try:
            return await get_internal_json(
                path=path, query_params=query_params, **self._common_kwargs(), **kwargs,
            )
        except InternalServiceError as exc:
            self._handle_error(exc)

    async def _post(
        self,
        path: str,
        json_body: dict,
        *,
        query_params: dict | list[tuple[str, object]] | None = None,
        **kwargs: Any,
    ) -> dict | None:
        try:
            return await post_internal_json(
                path=path, json_body=json_body, query_params=query_params,
                **self._common_kwargs(), **kwargs,
            )
        except InternalServiceError as exc:
            self._handle_error(exc)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        query_params: dict | list[tuple[str, object]] | None = None,
        **kwargs: Any,
    ) -> dict | None:
        try:
            return await request_internal_json(
                method=method, path=path, json_body=json_body,
                query_params=query_params, **self._common_kwargs(), **kwargs,
            )
        except InternalServiceError as exc:
            self._handle_error(exc)
