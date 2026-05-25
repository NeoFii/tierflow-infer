"""POST /internal/v1/classify — difficulty classification endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import (
    get_config_manager,
    get_engine,
    require_inference_secret,
)
from app.schema.classify import ClassifyRequest, ClassifyResponse
from app.service.classify_service import ClassifyService

router = APIRouter()


@router.post("/internal/v1/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    _secret: str = Depends(require_inference_secret),
    engine=Depends(get_engine),
    config_manager=Depends(get_config_manager),
) -> ClassifyResponse:
    return await ClassifyService.classify(request, engine, config_manager)
