"""Shared observability helpers for service entrypoints and internal calls."""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import threading
import uuid
from collections import deque
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-Id"

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id_var: ContextVar[str | None] = ContextVar("span_id", default=None)
_uid_var: ContextVar[str | None] = ContextVar("uid", default=None)

_ring_buffer: "RingBufferHandler | None" = None

_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(password\s*[:=]\s*)([^\s,&]+)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,&]+)"),
    re.compile(r"(?i)(token\s*[:=]\s*)([^\s,&]+)"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s,&]+)"),
    re.compile(r"(?i)(bearer\s+)(sk-[A-Za-z0-9._-]+|[A-Za-z0-9._-]{8,})"),
    re.compile(r"sk-[A-Za-z0-9._-]{8,}"),
]


def _redact(text: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        def _replace(match: re.Match[str]) -> str:
            if match.lastindex and match.lastindex >= 2:
                return f"{match.group(1)}[REDACTED]"
            return "[REDACTED]"
        text = pattern.sub(_replace, text)
    return text


def _resolve_component(name: str) -> str:
    """Resolve the component identifier for a logger name.

    Walks up the logger hierarchy to find the nearest _component attribute.
    Falls back to the raw logger name if none is found.
    """
    candidate = logging.getLogger(name)
    if hasattr(candidate, "_component"):
        return candidate._component
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        candidate = logging.getLogger(".".join(parts[:i]))
        if hasattr(candidate, "_component"):
            return candidate._component
    return name


def _build_entry(
    record: logging.LogRecord,
    service_name: str,
    env: str,
) -> dict[str, Any]:
    structured = getattr(record, "structured_fields", None)
    fields = dict(structured) if isinstance(structured, dict) else {}
    event = fields.pop("event", getattr(record, "event", "log"))
    service = fields.pop("service", service_name)
    component = fields.pop("component", _resolve_component(record.name))

    entry: dict[str, Any] = {
        "timestamp": datetime.fromtimestamp(record.created, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "level": record.levelname,
        "service": service,
        "component": component,
        "requestId": fields.pop("requestId", get_request_id()),
        "event": event,
    }

    # discard unused trace fields
    fields.pop("traceId", None)
    fields.pop("spanId", None)

    if structured is None:
        entry["message"] = _redact(record.getMessage())
    elif "message" in fields:
        entry["message"] = _redact(str(fields.pop("message")))
    else:
        fields.pop("message", None)

    uid = fields.pop("uid", get_uid())
    if uid:
        entry["uid"] = uid

    entry.update(fields)

    if record.exc_info:
        entry["exception"] = logging.Formatter().formatException(record.exc_info)

    return entry


class RingBufferHandler(logging.Handler):
    """In-memory ring buffer that stores recent structured log entries."""

    def __init__(self, capacity: int = 2000, *, service_name: str = "service", env: str = "development") -> None:
        super().__init__()
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._seq = 0
        self._lock = threading.Lock()
        self._service_name = service_name
        self._env = env

    def emit(self, record: logging.LogRecord) -> None:
        entry = _build_entry(record, self._service_name, self._env)
        with self._lock:
            self._seq += 1
            entry["seq"] = self._seq
            self._buf.append(entry)

    def snapshot(
        self,
        *,
        after_seq: int = 0,
        level: str | None = None,
        component: str | None = None,
        since: str | None = None,
        until: str | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int, int]:
        with self._lock:
            latest_seq = self._seq
            items = list(self._buf)
        if after_seq:
            items = [e for e in items if e["seq"] > after_seq]
        if level:
            allowed = _levels_at_or_above(level.upper())
            items = [e for e in items if e.get("level") in allowed]
        if component:
            items = [e for e in items if e.get("component", "").startswith(component)]
        if since:
            items = [e for e in items if e.get("timestamp", "") >= since]
        if until:
            items = [e for e in items if e.get("timestamp", "") <= until]
        if search:
            kw = search.lower()
            items = [
                e for e in items
                if kw in str(e.get("message", "")).lower()
                or kw in str(e.get("event", "")).lower()
                or kw in str(e.get("component", "")).lower()
            ]
        items.reverse()
        total = len(items)
        start = (page - 1) * page_size
        page_items = items[start : start + page_size]
        return page_items, total, latest_seq


_LEVEL_ORDER = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _levels_at_or_above(level: str) -> set[str]:
    try:
        idx = _LEVEL_ORDER.index(level)
    except ValueError:
        return set(_LEVEL_ORDER)
    return set(_LEVEL_ORDER[idx:])


def get_ring_buffer() -> RingBufferHandler | None:
    return _ring_buffer


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonLogFormatter(logging.Formatter):
    """Format log records as a single JSON object with the standard log schema."""

    def __init__(self, *, service_name: str, env: str = "development") -> None:
        super().__init__()
        self.service_name = service_name
        self.env = env

    def format(self, record: logging.LogRecord) -> str:
        entry = _build_entry(record, self.service_name, self.env)
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def configure_logging(
    log_level: str,
    *,
    service_name: str = "service",
    env: str = "development",
    log_dir: str | None = None,
    enable_file_logging: bool = False,
    file_prefix: str | None = None,
    file_max_bytes: int = 50 * 1024 * 1024,
    file_backup_count: int = 5,
    ring_buffer_capacity: int = 2000,
    force: bool = True,
) -> None:
    """Configure process-wide structured JSON logging."""
    global _ring_buffer
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    formatter = JsonLogFormatter(service_name=service_name, env=env)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if enable_file_logging and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or service_name
        handlers.append(
            RotatingFileHandler(
                Path(log_dir) / f"{prefix}.log",
                maxBytes=file_max_bytes,
                backupCount=file_backup_count,
                encoding="utf-8",
            )
        )

    if _ring_buffer is None:
        _ring_buffer = RingBufferHandler(capacity=ring_buffer_capacity, service_name=service_name, env=env)
        _ring_buffer.setLevel(level)
    handlers.append(_ring_buffer)

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    root = logging.getLogger()
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    logging.captureWarnings(True)


def parse_bool_env(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging_from_settings(settings: Any) -> None:
    configure_logging(
        getattr(settings, "LOG_LEVEL", "INFO"),
        service_name=getattr(settings, "SERVICE_NAME", "service"),
        env=getattr(settings, "ENV", "development"),
        log_dir=getattr(settings, "LOG_DIR", None),
        enable_file_logging=parse_bool_env(getattr(settings, "LOG_TO_FILE", False)),
        file_prefix=getattr(settings, "LOG_FILE_PREFIX", None),
        file_max_bytes=int(getattr(settings, "LOG_FILE_MAX_BYTES", 50 * 1024 * 1024)),
        file_backup_count=int(getattr(settings, "LOG_FILE_BACKUP_COUNT", 5)),
        ring_buffer_capacity=int(getattr(settings, "LOG_RING_BUFFER_CAPACITY", 2000)),
    )


def get_logger(component: str) -> logging.Logger:
    """Return a logger tagged with an explicit component identifier.

    Args:
        component: Business component name in "domain.module" format.
                   Use constants from app.common.components.C.
    """
    logger = logging.getLogger(component)
    if not hasattr(logger, "_component"):
        logger._component = component  # type: ignore[attr-defined]
    return logger


def set_request_id(request_id: str) -> Token:
    return _request_id_var.set(request_id)

def get_request_id() -> str | None:
    return _request_id_var.get()

def reset_request_id(token: Token) -> None:
    _request_id_var.reset(token)

def set_trace_id(trace_id: str) -> Token:
    return _trace_id_var.set(trace_id)

def get_trace_id() -> str | None:
    return _trace_id_var.get()

def reset_trace_id(token: Token) -> None:
    _trace_id_var.reset(token)

def set_span_id(span_id: str) -> Token:
    return _span_id_var.set(span_id)

def get_span_id() -> str | None:
    return _span_id_var.get()

def reset_span_id(token: Token) -> None:
    _span_id_var.reset(token)

def set_uid(uid: str) -> Token:
    return _uid_var.set(uid)

def get_uid() -> str | None:
    return _uid_var.get()

def reset_uid(token: Token) -> None:
    _uid_var.reset(token)


def _json_default(value: Any) -> str:
    return str(value)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured JSON log event."""
    exc_info = fields.pop("exc_info", None)
    error_code = fields.pop("errorCode", None)
    error_detail = fields.pop("errorDetail", None)
    if error_code:
        fields["error"] = {"code": error_code, "detail": error_detail or ""}
    logger.log(
        level,
        event,
        extra={"structured_fields": {"event": event, **fields}},
        exc_info=exc_info,
    )


def _safe_int(value: Any) -> int | None:
    """Cast a Content-Length-like header to int, returning None if absent/invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_access_extra(request: Request, response: Response | None) -> dict[str, Any]:
    """Collect contextual fields the access middleware tags onto every request log.

    Reads:
    - request.scope["route"] for the FastAPI-matched route pattern (no path
      param noise — `/users/{uid}` instead of `/users/u123`)
    - request.state.api_key_prefix (set by router-service api-key dependency)
    - _uid_var (already populated for authenticated requests)
    - request/response Content-Length, User-Agent, query string (redacted)
    """
    fields: dict[str, Any] = {}

    matched = request.scope.get("route") if isinstance(request.scope, dict) else None
    route_path = getattr(matched, "path", None)
    if route_path:
        fields["routeName"] = route_path

    user_agent = request.headers.get("user-agent")
    if user_agent:
        # User agents can be long; cap at 256 chars to keep the ring buffer tidy.
        fields["userAgent"] = user_agent[:256]

    request_bytes = _safe_int(request.headers.get("content-length"))
    if request_bytes is not None:
        fields["requestBytes"] = request_bytes

    if response is not None:
        response_bytes = _safe_int(response.headers.get("content-length"))
        if response_bytes is not None:
            fields["responseBytes"] = response_bytes

    if request.url.query:
        fields["query"] = _redact(request.url.query)[:512]

    api_key_prefix = getattr(request.state, "api_key_prefix", None)
    if api_key_prefix:
        fields["apiKeyPrefix"] = str(api_key_prefix)[:16]

    user_id = get_uid()
    if user_id:
        # set_uid was already called by an upstream auth dep; mirror to userId for
        # readability in the admin "服务日志" panel (uid is also shown via the
        # ContextVar route).
        fields["userId"] = str(user_id)

    return fields


def install_observability(app: FastAPI, *, service_name: str) -> None:
    """Install request-id, trace-id, span-id propagation and structured access logging."""
    app.state.service_name = service_name
    access_logger = logging.getLogger(f"{service_name}.access")

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        trace_id = request.headers.get(TRACE_ID_HEADER) or uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:12]

        rid_token = set_request_id(request_id)
        tid_token = set_trace_id(trace_id)
        sid_token = set_span_id(span_id)
        request.state.request_id = request_id
        started_at = perf_counter()
        client_ip = request.client.host if request.client else None

        try:
            response = await call_next(request)
        except Exception as exc:
            durationMs = round((perf_counter() - started_at) * 1000, 2)
            exception_handler = app.exception_handlers.get(Exception)
            if exception_handler is not None:
                response = exception_handler(request, exc)
                if inspect.isawaitable(response):
                    response = await response
                if not isinstance(response, Response):
                    response = JSONResponse(
                        status_code=500,
                        content={"detail": "Internal Server Error"},
                    )
            else:
                log_event(
                    logging.getLogger("common.exceptions"),
                    logging.ERROR,
                    "unhandledException",
                    message=f"未捕获异常: {type(exc).__name__}",
                    method=request.method,
                    path=request.url.path,
                    statusCode=500,
                    errorCode=type(exc).__name__,
                    errorDetail=str(exc),
                    exc_info=True,
                )
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "Internal Server Error"},
                )
            response.headers.setdefault(REQUEST_ID_HEADER, request_id)
            response.headers.setdefault(TRACE_ID_HEADER, trace_id)
            log_event(
                access_logger,
                logging.ERROR,
                "requestError",
                message=f"{request.method} {request.url.path} {response.status_code} {durationMs}ms",
                method=request.method,
                path=request.url.path,
                statusCode=response.status_code,
                durationMs=durationMs,
                clientIp=client_ip,
                errorCode=type(exc).__name__,
                errorDetail=str(exc),
                **_build_access_extra(request, response),
            )
            reset_span_id(sid_token)
            reset_trace_id(tid_token)
            reset_request_id(rid_token)
            return response

        durationMs = round((perf_counter() - started_at) * 1000, 2)
        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
        response.headers.setdefault(TRACE_ID_HEADER, trace_id)
        log_event(
            access_logger,
            logging.INFO,
            "requestComplete",
            message=f"{request.method} {request.url.path} {response.status_code} {durationMs}ms",
            method=request.method,
            path=request.url.path,
            statusCode=response.status_code,
            durationMs=durationMs,
            clientIp=client_ip,
            **_build_access_extra(request, response),
        )
        reset_span_id(sid_token)
        reset_trace_id(tid_token)
        reset_request_id(rid_token)
        return response
