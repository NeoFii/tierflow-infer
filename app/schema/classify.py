"""Pydantic schemas for the /internal/v1/classify endpoint."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ClassifyRequest(BaseModel):
    messages: list[Dict[str, Any]] = Field(..., min_length=1, max_length=512)
    profile_id: Optional[str] = Field(default=None, max_length=64)
    request_id: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[a-zA-Z0-9_\-]*$"
    )

    @model_validator(mode="after")
    def _check_message_content_size(self) -> ClassifyRequest:
        for msg in self.messages:
            content = str(msg.get("content", ""))
            if len(content) > 100_000:
                raise ValueError("single message content exceeds 100KB limit")
        return self


class ClassifyResponse(BaseModel):
    request_id: str
    scores_0_2: Dict[str, float]
    proto_weighted_0_2: Optional[float] = None
    total_score_0_10: float
    score_source: str
    latency_ms: Optional[float] = None
