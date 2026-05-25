"""FastAPI dependency injection functions for tierflow-infer."""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from fastapi import Header

from app.common.components import C
from app.common.observability import get_logger
from app.core.config import get_settings
from app.core.exceptions import InferenceAuthError, InferenceUnavailableError

if TYPE_CHECKING:
    from app.service.config_manager import ConfigManager
    from app.service.router_engine import HybridIntegratedDifficultyRouter

logger = get_logger(C.CORE_DEPS)

_engine: HybridIntegratedDifficultyRouter | None = None
_config_manager: ConfigManager | None = None


def set_engine(engine: HybridIntegratedDifficultyRouter) -> None:
    global _engine
    _engine = engine


def set_config_manager(cm: ConfigManager) -> None:
    global _config_manager
    _config_manager = cm


def get_engine() -> HybridIntegratedDifficultyRouter:
    if _engine is None:
        raise InferenceUnavailableError("router engine not initialized")
    return _engine


def get_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise InferenceUnavailableError("config manager not initialized")
    return _config_manager


def require_inference_secret(
    x_inference_secret: str | None = Header(default=None),
) -> str:
    settings = get_settings()
    expected = settings.INFERENCE_SERVICE_SECRET
    if not expected:
        if settings.INFERENCE_ALLOW_INSECURE_DEV and settings.ENV != "production":
            logger.warning(
                "INFERENCE_SERVICE_SECRET 未设置 — classify 端点无保护 (开发模式)"
            )
            return ""
        if settings.INFERENCE_ALLOW_INSECURE_DEV and settings.ENV == "production":
            logger.error(
                "INFERENCE_ALLOW_INSECURE_DEV 在生产环境中启用 — 拒绝绕过认证"
            )
        raise InferenceUnavailableError("推理服务未配置")
    if not x_inference_secret or not hmac.compare_digest(
        x_inference_secret.encode("utf-8"), expected.encode("utf-8")
    ):
        raise InferenceAuthError("forbidden")
    return x_inference_secret
