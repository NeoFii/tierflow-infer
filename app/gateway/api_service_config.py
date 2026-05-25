"""Gateway for fetching routing configuration from tierflow-core."""

from __future__ import annotations

from app.common.components import C
from app.common.gateway.base import BaseGateway
from app.common.internal import InternalCircuitOpenError, InternalServiceUnavailableError
from app.common.observability import get_logger
from app.core.config import get_settings


logger = get_logger(C.GATEWAY_API)


class ApiServiceConfigGateway(BaseGateway):
    """Fetch active routing config (inference view) from tierflow-core."""

    def __init__(self) -> None:
        settings = get_settings()
        super().__init__(
            "tierflow-core",
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
            logger.warning("tierflow-core 不可用，将使用回退配置", exc_info=True)
            return None
