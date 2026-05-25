"""Gateway for fetching routing configuration from api-service."""

from __future__ import annotations

from app.common.components import C
from app.common.gateway.base import BaseGateway
from app.common.internal import InternalCircuitOpenError, InternalServiceUnavailableError
from app.common.observability import get_logger
from app.core.config import get_settings


logger = get_logger(C.GATEWAY_API)


class ApiServiceConfigGateway(BaseGateway):
    """Fetch active routing config (inference view) from api-service."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            "api-service",
            base_url=settings.API_SERVICE_URL,
            timeout=settings.CONFIG_FETCH_TIMEOUT_SECONDS,
        )

    async def fetch_active_config(self) -> dict | None:
        try:
            return await self._get(
                "/api/v1/internal/routing-config/active/inference",
                allow_404=True,
            )
        except (InternalServiceUnavailableError, InternalCircuitOpenError):
            logger.warning("api-service unavailable, will use fallback", exc_info=True)
            return None
