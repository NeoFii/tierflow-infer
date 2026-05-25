"""ConfigManager: tierflow-core config with cached_previous fallback."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from app.common.components import C
from app.common.internal import InternalServiceResponseError
from app.common.observability import get_logger, log_event
from app.core.exceptions import InferenceConfigError
from app.gateway.api_service_config import ApiServiceConfigGateway
from app.utils.runtime_config import normalize_inference_config, normalize_profile_config

_logger = get_logger(C.CLASSIFY_CONFIG)


class ConfigManager:
    """Manage routing configuration with tierflow-core as sole source.

    Supports multi-profile mode (D-05): maintains a dict[slug, config] mapping.
    """

    def __init__(
        self,
        *,
        gateway: ApiServiceConfigGateway,
        refresh_interval_seconds: int = 60,
    ) -> None:
        self._gateway = gateway
        self._refresh_interval = refresh_interval_seconds
        self._cached_config: Dict[str, Any] | None = None
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._config_version: int | None = None
        self._config_source: str = "none"
        self._last_updated_at: datetime | None = None
        self._refresh_task: asyncio.Task | None = None

    @property
    def config_version(self) -> int | None:
        return self._config_version

    @property
    def config_source(self) -> str:
        return self._config_source

    @property
    def last_updated_at(self) -> datetime | None:
        return self._last_updated_at

    async def start(self) -> None:
        try:
            admin_config = await self._gateway.fetch_active_config()
        except InternalServiceResponseError as exc:
            if exc.status_code in (401, 403):
                raise RuntimeError(
                    f"tierflow-core rejected credentials (HTTP {exc.status_code}): {exc.detail}"
                ) from exc
            raise RuntimeError(
                f"failed to fetch routing config from tierflow-core (HTTP {exc.status_code})"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "failed to fetch routing config from tierflow-core"
            ) from exc

        if admin_config is None:
            raise RuntimeError(
                "tierflow-core returned no active routing config"
            )

        self._apply_config(admin_config)
        log_event(
            _logger, logging.INFO, "config_loaded",
            message="路由配置已从 tierflow-core 加载",
            version=self._config_version,
            profileCount=len(self._profiles),
        )

        self._refresh_task = asyncio.create_task(self._poll_loop())

    def _apply_config(self, admin_config: Dict[str, Any]) -> None:
        """Parse admin config and update internal state atomically."""
        profiles_raw = admin_config.get("profiles")
        if profiles_raw and isinstance(profiles_raw, list):
            # Multi-profile format (D-05): {version, status, profiles: [...]}
            new_profiles: Dict[str, Dict[str, Any]] = {}
            for p in profiles_raw:
                slug = p.get("slug", "")
                if not slug:
                    continue
                new_profiles[slug] = normalize_profile_config(p)
            self._profiles = new_profiles
            # Backward compat: _cached_config points to first profile or None
            if new_profiles:
                first_slug = next(iter(new_profiles))
                self._cached_config = new_profiles[first_slug]
            else:
                self._cached_config = None
        else:
            # Legacy single-config format (backward compat)
            self._cached_config = normalize_inference_config(admin_config)
            slug = admin_config.get("slug", "default")
            self._profiles = {slug: self._cached_config}

        self._config_version = admin_config.get("version")
        self._config_source = "admin"
        self._last_updated_at = datetime.now(timezone.utc)

    async def stop(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)
            self._refresh_task = None

    def load(self) -> Dict[str, Any]:
        """Load the default/first cached config (deprecated, use load_profile)."""
        if self._cached_config is None:
            raise InferenceConfigError("config not loaded — ConfigManager not started")
        return self._cached_config

    def load_profile(self, profile_id: str) -> Dict[str, Any]:
        """Load config for a specific profile by slug.

        Raises InferenceConfigError if profile_id is unknown (T-11-10).
        """
        if profile_id not in self._profiles:
            raise InferenceConfigError(f"unknown profile: {profile_id}")
        return self._profiles[profile_id]

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval)
            try:
                resp = await self._gateway.fetch_active_config()
                if resp is None:
                    if self._config_source == "admin":
                        self._config_source = "cached_previous"
                        _logger.warning("tierflow-core 配置不可用，使用缓存配置")
                    continue
                new_version = resp.get("version")
                if new_version != self._config_version:
                    log_event(
                        _logger, logging.INFO, "config_updated",
                        message="路由配置版本已更新",
                        oldVersion=self._config_version,
                        newVersion=new_version,
                    )
                self._apply_config(resp)
            except asyncio.CancelledError:
                raise
            except InternalServiceResponseError as exc:
                if exc.status_code in (401, 403):
                    _logger.error(
                        "HMAC 认证被拒绝 (HTTP %s)，使用缓存配置",
                        exc.status_code,
                    )
                else:
                    _logger.warning(
                        "配置刷新失败 (HTTP %s)，保持当前配置",
                        exc.status_code,
                        exc_info=True,
                    )
                if self._config_source == "admin":
                    self._config_source = "cached_previous"
            except Exception:
                if self._config_source == "admin":
                    self._config_source = "cached_previous"
                _logger.warning("配置刷新失败，保持当前配置", exc_info=True)
