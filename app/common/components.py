"""Centralized component identifiers for tierflow-infer structured logging."""

from __future__ import annotations


class C:
    # ── classify 域 ───────────────────────────────────────────
    CLASSIFY_SERVICE = "classify.service"
    CLASSIFY_ENGINE = "classify.engine"

    # ── core 域 ───────────────────────────────────────────────
    CORE_MAIN = "core.main"
    CORE_DEPS = "core.deps"
    CORE_EXCEPTIONS = "core.exceptions"
