"""Centralized component identifiers for inference-service structured logging."""

from __future__ import annotations


class C:
    # ── classify 域 ───────────────────────────────────────────
    CLASSIFY_SERVICE = "classify.service"
    CLASSIFY_ENGINE = "classify.engine"
    CLASSIFY_CONFIG = "classify.config"

    # ── gateway 域 ────────────────────────────────────────────
    GATEWAY_ADMIN = "gateway.admin"
    GATEWAY_API = "gateway.api"

    # ── core 域 ───────────────────────────────────────────────
    CORE_MAIN = "core.main"
    CORE_DEPS = "core.deps"
    CORE_EXCEPTIONS = "core.exceptions"

    # ── util ──────────────────────────────────────────────────
    UTIL_RUNTIME_CONFIG = "util.runtime_config"
