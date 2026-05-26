"""ClassifyService: orchestrates engine for classification scoring."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from app.common.components import C
from app.common.observability import get_logger, get_request_id, log_event
from app.schema.classify import ClassifyResponse

if TYPE_CHECKING:
    from app.schema.classify import ClassifyRequest
    from app.service.router_engine import HybridIntegratedDifficultyRouter

logger = get_logger(C.CLASSIFY_SERVICE)

CLASSIFY_SOFT_TIMEOUT_SECONDS = 30.0

_gpu_semaphore: asyncio.Semaphore | None = None


def init_gpu_semaphore(limit: int) -> None:
    global _gpu_semaphore
    _gpu_semaphore = asyncio.Semaphore(limit)


class ClassifyService:

    @staticmethod
    async def classify(
        request: ClassifyRequest,
        engine: HybridIntegratedDifficultyRouter,
    ) -> ClassifyResponse:
        request_id = get_request_id() or request.request_id or ""
        messages_count = len(request.messages)

        queued_ms = 0.0
        if _gpu_semaphore is not None:
            t_queue = time.monotonic()
            await _gpu_semaphore.acquire()
            queued_ms = round((time.monotonic() - t_queue) * 1000, 2)
        try:
            t_start = time.monotonic()
            try:
                result = await asyncio.to_thread(
                    engine.predict_chat_messages,
                    request.messages,
                    request_id=request_id,
                )
            except Exception as exc:
                engine_latency_ms = round((time.monotonic() - t_start) * 1000, 2)
                log_event(
                    logger, logging.ERROR, "classify_failed",
                    message="分类推理失败",
                    requestId=request_id,
                    profileId=request.profile_id,
                    messagesCount=messages_count,
                    queuedMs=queued_ms,
                    engineLatencyMs=engine_latency_ms,
                    errorCode=type(exc).__name__,
                    errorDetail=str(exc)[:256],
                )
                raise
            elapsed = time.monotonic() - t_start
        finally:
            if _gpu_semaphore is not None:
                _gpu_semaphore.release()

        latency_ms = round(elapsed * 1000, 2)
        soft_timeout = elapsed > CLASSIFY_SOFT_TIMEOUT_SECONDS

        log_event(
            logger, logging.INFO, "classify_complete",
            message=f"分类完成 score={result['total_score_0_10']:.1f} {latency_ms}ms",
            requestId=request_id,
            profileId=request.profile_id,
            messagesCount=messages_count,
            queuedMs=queued_ms,
            engineLatencyMs=latency_ms,
            totalScore=result["total_score_0_10"],
            scoreSource=result["score_source"],
            protoWeighted_0_2=result.get("proto_weighted_0_2"),
            softTimeout=soft_timeout,
        )

        return ClassifyResponse(
            request_id=request_id,
            scores_0_2=result["scores_0_2"],
            proto_weighted_0_2=result.get("proto_weighted_0_2"),
            total_score_0_10=result["total_score_0_10"],
            score_source=result["score_source"],
            latency_ms=latency_ms,
        )
