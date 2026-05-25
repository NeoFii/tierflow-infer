"""Shared helpers for service-to-service contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import hmac
import json
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from fastapi import Header, HTTPException, Request, status

from app.common.observability import REQUEST_ID_HEADER, get_request_id, TRACE_ID_HEADER, get_trace_id

INTERNAL_CALLER_HEADER = "X-Internal-Service"
INTERNAL_TIMESTAMP_HEADER = "X-Internal-Timestamp"
INTERNAL_SIGNATURE_HEADER = "X-Internal-Signature"
INVALID_INTERNAL_SECRET_DETAIL = "Invalid internal secret"
INVALID_INTERNAL_CALLER_DETAIL = "Invalid internal caller"


class InternalServiceError(httpx.HTTPError):
    """Base exception for internal service transport and availability failures."""

    def __init__(
        self,
        message: str,
        *,
        target_service: str,
        path: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.target_service = target_service
        self.path = path
        self.status_code = status_code


class InternalServiceUnavailableError(InternalServiceError):
    """Raised when an internal service is unavailable after retries."""


class InternalCircuitOpenError(InternalServiceError):
    """Raised when the local circuit breaker is open for a target service."""


class InternalServiceResponseError(InternalServiceError):
    """Raised when an internal service returns a non-retriable 4xx response."""

    def __init__(
        self,
        message: str,
        *,
        target_service: str,
        path: str,
        status_code: int,
        detail: str | None = None,
    ) -> None:
        super().__init__(
            message,
            target_service=target_service,
            path=path,
            status_code=status_code,
        )
        self.detail = detail


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_until: float | None = None


_CIRCUIT_BREAKERS: dict[str, _CircuitState] = {}


def reset_internal_circuit_breakers() -> None:
    """Clear in-memory circuit-breaker state, primarily for tests."""
    _CIRCUIT_BREAKERS.clear()


def _canonicalize_json_body(json_body: dict | list | None) -> str:
    if json_body is None:
        return ""
    return json.dumps(json_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonicalize_request_body(body: bytes) -> str:
    if not body:
        return ""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="ignore")
    return _canonicalize_json_body(parsed)


def _canonicalize_query_pairs(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    normalized = sorted((str(key), str(value)) for key, value in pairs)
    return urlencode(normalized, doseq=True)


def _canonicalize_request_query(raw_query: str) -> str:
    if not raw_query:
        return ""
    return _canonicalize_query_pairs(parse_qsl(raw_query, keep_blank_values=True))


def _canonicalize_query_params(
    query_params: dict | list[tuple[str, object]] | None,
) -> str:
    if not query_params:
        return ""

    pairs: list[tuple[str, str]] = []
    items = query_params.items() if isinstance(query_params, dict) else query_params
    for key, value in items:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for entry in value:
                if entry is None:
                    continue
                pairs.append((str(key), str(entry)))
            continue
        pairs.append((str(key), str(value)))
    return _canonicalize_query_pairs(pairs)


def _canonicalize_request_target(
    path: str,
    *,
    raw_query: str = "",
    query_params: dict | list[tuple[str, object]] | None = None,
) -> str:
    split = urlsplit(path)
    normalized_path = split.path or path or "/"
    query = _canonicalize_request_query(raw_query or split.query)
    if query_params:
        query = _canonicalize_query_params(query_params)
    if query:
        return f"{normalized_path}?{query}"
    return normalized_path


def _build_internal_signature(
    *,
    secret: str,
    caller_service: str,
    method: str,
    request_target: str,
    timestamp: str,
    canonical_body: str,
) -> str:
    body_digest = hashlib.sha256(canonical_body.encode("utf-8")).hexdigest()
    payload = f"{caller_service}|{method.upper()}|{request_target}|{timestamp}|{body_digest}"
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def build_internal_auth_dependency(
    expected_secret: str,
    *,
    request_ttl_seconds: int = 30,
    allowed_callers: set[str] | None = None,
):
    """Create a dependency that validates signed internal service calls."""

    async def verify_internal_secret(
        request: Request,
        x_internal_service: str | None = Header(None, alias=INTERNAL_CALLER_HEADER),
        x_internal_timestamp: str | None = Header(None, alias=INTERNAL_TIMESTAMP_HEADER),
        x_internal_signature: str | None = Header(None, alias=INTERNAL_SIGNATURE_HEADER),
    ) -> None:
        if x_internal_service and x_internal_timestamp and x_internal_signature:
            try:
                timestamp_value = int(x_internal_timestamp)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=INVALID_INTERNAL_SECRET_DETAIL,
                ) from exc
            if abs(int(time.time()) - timestamp_value) > request_ttl_seconds:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=INVALID_INTERNAL_SECRET_DETAIL,
                )
            canonical_body = _canonicalize_request_body(await request.body())
            request_target = _canonicalize_request_target(
                request.url.path,
                raw_query=request.url.query,
            )
            expected_signature = _build_internal_signature(
                secret=expected_secret,
                caller_service=x_internal_service,
                method=request.method,
                request_target=request_target,
                timestamp=x_internal_timestamp,
                canonical_body=canonical_body,
            )
            if not hmac.compare_digest(expected_signature, x_internal_signature):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=INVALID_INTERNAL_SECRET_DETAIL,
                )
            if allowed_callers and x_internal_service not in allowed_callers:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=INVALID_INTERNAL_CALLER_DETAIL,
                )
            request.state.internal_caller = x_internal_service
            return

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=INVALID_INTERNAL_SECRET_DETAIL,
        )

    return verify_internal_secret


def build_internal_headers(
    *,
    secret: str,
    caller_service: str,
    method: str,
    path: str,
    json_body: dict | None = None,
    query_params: dict | list[tuple[str, object]] | None = None,
) -> dict[str, str]:
    """Build the canonical headers for internal service calls."""
    timestamp = str(int(time.time()))
    canonical_body = _canonicalize_json_body(json_body)
    request_target = _canonicalize_request_target(path, query_params=query_params)
    headers = {
        INTERNAL_CALLER_HEADER: caller_service,
        INTERNAL_TIMESTAMP_HEADER: timestamp,
        INTERNAL_SIGNATURE_HEADER: _build_internal_signature(
            secret=secret,
            caller_service=caller_service,
            method=method,
            request_target=request_target,
            timestamp=timestamp,
            canonical_body=canonical_body,
        ),
    }
    request_id = get_request_id()
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    trace_id = get_trace_id()
    if trace_id:
        headers[TRACE_ID_HEADER] = trace_id
    return headers


def _build_internal_url(base_url: str, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url.rstrip('/')}{normalized_path}"


def _circuit_key(target_service: str, base_url: str) -> str:
    return f"{target_service}|{base_url.rstrip('/')}"


def _get_circuit_state(key: str) -> _CircuitState:
    state = _CIRCUIT_BREAKERS.get(key)
    if state is None:
        state = _CircuitState()
        _CIRCUIT_BREAKERS[key] = state
    return state


def _check_circuit_open(*, key: str, target_service: str, path: str) -> None:
    state = _get_circuit_state(key)
    opened_until = state.opened_until
    if opened_until and opened_until > time.time():
        raise InternalCircuitOpenError(
            f"{target_service} circuit is open",
            target_service=target_service,
            path=path,
        )
    if opened_until and opened_until <= time.time():
        state.opened_until = None


def _record_success(key: str) -> None:
    state = _get_circuit_state(key)
    state.consecutive_failures = 0
    state.opened_until = None


def _record_failure(
    *,
    key: str,
    threshold: int,
    cooldown_seconds: float,
) -> None:
    state = _get_circuit_state(key)
    state.consecutive_failures += 1
    if threshold > 0 and state.consecutive_failures >= threshold:
        state.opened_until = time.time() + max(cooldown_seconds, 0.0)


# ---------------------------------------------------------------------------
# Shared httpx client pool (connection reuse across requests)
# ---------------------------------------------------------------------------

_HTTP_CLIENTS: dict[str, httpx.AsyncClient] = {}


def get_internal_client(base_url: str, *, timeout: float = 10.0) -> httpx.AsyncClient:
    key = base_url.rstrip("/")
    if key not in _HTTP_CLIENTS:
        _HTTP_CLIENTS[key] = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _HTTP_CLIENTS[key]


async def close_internal_clients() -> None:
    """Gracefully close all pooled HTTP clients. Call from app shutdown."""
    for client in _HTTP_CLIENTS.values():
        await client.aclose()
    _HTTP_CLIENTS.clear()


def _extract_error_detail(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if detail is not None:
            return str(detail)

    text = getattr(response, "text", "")
    if text:
        return text.strip() or None
    return None


async def request_internal_json(
    *,
    method: str,
    base_url: str,
    target_service: str,
    path: str,
    secret: str,
    caller_service: str,
    timeout: float,
    json_body: dict | None = None,
    query_params: dict | list[tuple[str, object]] | None = None,
    allow_404: bool = False,
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.2,
    circuit_breaker_threshold: int = 3,
    circuit_breaker_cooldown_seconds: float = 30.0,
) -> dict | None:
    """Perform a canonical internal request and decode JSON."""
    attempts = max(max_retries, 0) + 1
    url = _build_internal_url(base_url, path)
    breaker_key = _circuit_key(target_service, base_url)

    _check_circuit_open(key=breaker_key, target_service=target_service, path=path)

    for attempt in range(attempts):
        try:
            client = get_internal_client(base_url, timeout=timeout)
            response = await client.request(
                method,
                url,
                headers=build_internal_headers(
                    secret=secret,
                    caller_service=caller_service,
                    method=method,
                    path=path,
                    json_body=json_body,
                    query_params=query_params,
                ),
                json=json_body,
                params=query_params,
            )
        except httpx.HTTPError as exc:
            if attempt == attempts - 1:
                _record_failure(
                    key=breaker_key,
                    threshold=circuit_breaker_threshold,
                    cooldown_seconds=circuit_breaker_cooldown_seconds,
                )
                raise InternalServiceUnavailableError(
                    f"{target_service} request failed",
                    target_service=target_service,
                    path=path,
                ) from exc
            await asyncio.sleep(retry_backoff_seconds * (attempt + 1))
            continue

        if allow_404 and response.status_code == status.HTTP_404_NOT_FOUND:
            _record_success(breaker_key)
            return None
        if response.status_code >= 500 and attempt < attempts - 1:
            await asyncio.sleep(retry_backoff_seconds * (attempt + 1))
            continue
        if response.status_code >= 500:
            _record_failure(
                key=breaker_key,
                threshold=circuit_breaker_threshold,
                cooldown_seconds=circuit_breaker_cooldown_seconds,
            )
            raise InternalServiceUnavailableError(
                f"{target_service} returned {response.status_code}",
                target_service=target_service,
                path=path,
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            _record_success(breaker_key)
            detail = _extract_error_detail(response)
            raise InternalServiceResponseError(
                f"{target_service} returned {response.status_code}",
                target_service=target_service,
                path=path,
                status_code=response.status_code,
                detail=detail,
            )
        _record_success(breaker_key)
        try:
            return response.json()
        except ValueError as exc:
            _record_failure(
                key=breaker_key,
                threshold=circuit_breaker_threshold,
                cooldown_seconds=circuit_breaker_cooldown_seconds,
            )
            raise InternalServiceUnavailableError(
                f"{target_service} returned invalid JSON",
                target_service=target_service,
                path=path,
                status_code=response.status_code,
            ) from exc

    raise RuntimeError("Internal request retry loop exited unexpectedly")


async def get_internal_json(
    *,
    base_url: str,
    target_service: str,
    path: str,
    secret: str,
    caller_service: str,
    timeout: float,
    query_params: dict | list[tuple[str, object]] | None = None,
    allow_404: bool = False,
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.2,
    circuit_breaker_threshold: int = 3,
    circuit_breaker_cooldown_seconds: float = 30.0,
) -> dict | None:
    """Perform a canonical internal GET request and decode JSON."""
    return await request_internal_json(
        method="GET",
        base_url=base_url,
        target_service=target_service,
        path=path,
        secret=secret,
        caller_service=caller_service,
        timeout=timeout,
        query_params=query_params,
        allow_404=allow_404,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=circuit_breaker_cooldown_seconds,
    )


async def post_internal_json(
    *,
    base_url: str,
    target_service: str,
    path: str,
    secret: str,
    caller_service: str,
    timeout: float,
    json_body: dict,
    query_params: dict | list[tuple[str, object]] | None = None,
    allow_404: bool = False,
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.2,
    circuit_breaker_threshold: int = 3,
    circuit_breaker_cooldown_seconds: float = 30.0,
) -> dict | None:
    """Perform a canonical internal POST request and decode JSON."""
    return await request_internal_json(
        method="POST",
        base_url=base_url,
        target_service=target_service,
        path=path,
        secret=secret,
        caller_service=caller_service,
        timeout=timeout,
        json_body=json_body,
        query_params=query_params,
        allow_404=allow_404,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=circuit_breaker_cooldown_seconds,
    )


async def patch_internal_json(
    *,
    base_url: str,
    target_service: str,
    path: str,
    secret: str,
    caller_service: str,
    timeout: float,
    json_body: dict,
    query_params: dict | list[tuple[str, object]] | None = None,
    allow_404: bool = False,
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.2,
    circuit_breaker_threshold: int = 3,
    circuit_breaker_cooldown_seconds: float = 30.0,
) -> dict | None:
    """Perform a canonical internal PATCH request and decode JSON."""
    return await request_internal_json(
        method="PATCH",
        base_url=base_url,
        target_service=target_service,
        path=path,
        secret=secret,
        caller_service=caller_service,
        timeout=timeout,
        json_body=json_body,
        query_params=query_params,
        allow_404=allow_404,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_cooldown_seconds=circuit_breaker_cooldown_seconds,
    )
