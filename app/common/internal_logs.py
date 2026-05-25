"""Shared internal log-reading endpoint for all services."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.common.observability import get_ring_buffer


def build_internal_logs_router(auth_dependency) -> APIRouter:
    """Return a router with GET /internal/logs backed by the process ring buffer."""
    router = APIRouter(prefix="/internal", tags=["internal"])

    @router.get("/logs", summary="Read recent logs from ring buffer")
    async def read_logs(
        _: None = Depends(auth_dependency),
        level: str | None = Query(None, max_length=10),
        component: str | None = Query(None, max_length=50),
        since: str | None = Query(None, max_length=30),
        until: str | None = Query(None, max_length=30),
        search: str | None = Query(None, max_length=200),
        after_seq: int = Query(0, ge=0),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
    ):
        buf = get_ring_buffer()
        if buf is None:
            return {"service": "unknown", "entries": [], "total": 0, "latest_seq": 0}
        entries, total, latest_seq = buf.snapshot(
            after_seq=after_seq,
            level=level,
            component=component,
            since=since,
            until=until,
            search=search,
            page=page,
            page_size=page_size,
        )
        return {
            "service": buf._service_name,
            "entries": entries,
            "total": total,
            "latest_seq": latest_seq,
        }

    return router
