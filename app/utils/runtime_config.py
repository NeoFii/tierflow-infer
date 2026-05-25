"""Runtime configuration normalization and cloning."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from app.common.components import C
from app.common.observability import get_logger

logger = get_logger(C.UTIL_RUNTIME_CONFIG)

from app.core.config import (
    FIVEWAY_DEFAULT_WEIGHTS,
    FIVEWAY_ROUTE_ORDER,
)
from app.utils.scoring import parse_score_bands

DEFAULT_ROUTER_ALIAS = "auto"

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} references with environment variable values."""
    def _replace(m: re.Match) -> str:
        env_val = os.environ.get(m.group(1))
        if env_val is None:
            raise ValueError(
                f"environment variable {m.group(1)} is not set (referenced in runtime config)"
            )
        return env_val
    return _ENV_VAR_RE.sub(_replace, value)


def build_default_runtime_config() -> Dict[str, Any]:
    return {
        "router_alias": DEFAULT_ROUTER_ALIAS,
        "route_order": list(FIVEWAY_ROUTE_ORDER),
        "weights": dict(FIVEWAY_DEFAULT_WEIGHTS),
        "score_bands": "0-3:5,3-5:4,5-7:3,7-9:2,9-10:1",
        "tier_model_map": {
            "1": "gpt-5-4",
            "2": "minimax-m2-7",
            "3": "qwen-3-5-397b-a17b",
            "4": "qwen3-5-flash",
            "5": "GLM4.7-Flash",
        },
        "model_providers": {},
    }


def _validate_weights(weights_raw: Any, defaults: Dict[str, float]) -> Dict[str, float]:
    if isinstance(weights_raw, list):
        if len(weights_raw) != 5:
            raise ValueError("weights list must contain exactly 5 numbers")
        weights = dict(zip(FIVEWAY_ROUTE_ORDER, [float(x) for x in weights_raw]))
    elif isinstance(weights_raw, dict):
        weights = {}
        for name in FIVEWAY_ROUTE_ORDER:
            if name not in weights_raw:
                raise ValueError(f"weights is missing route: {name}")
            weights[name] = float(weights_raw[name])
    else:
        raise ValueError("weights must be a dict or a list")
    if any(value < 0 for value in weights.values()):
        raise ValueError("weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("weights sum must be greater than 0")
    return weights


def _validate_score_bands(
    score_bands_value: Any,
    score_bands_raw_hint: str = "",
) -> Tuple[List[Tuple[float, float, int]], str]:
    if isinstance(score_bands_value, list):
        score_bands: List[Tuple[float, float, int]] = []
        for item in score_bands_value:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("score_bands list items must be [start, end, tier]")
            score_bands.append((float(item[0]), float(item[1]), int(item[2])))
        if not score_bands:
            raise ValueError("score bands must not be empty")
        if score_bands_raw_hint:
            score_bands_raw = score_bands_raw_hint
        else:
            pieces = []
            for start, end, tier in score_bands:
                pieces.append(f"{start}-{end}:{tier}" if start != end else f"{start}:{tier}")
            score_bands_raw = ",".join(pieces)
    else:
        score_bands_raw = str(score_bands_value).strip()
        score_bands = parse_score_bands(score_bands_raw)
    return score_bands, score_bands_raw


def _validate_tier_model_map(tier_map_raw: Any) -> Dict[int, str]:
    if not isinstance(tier_map_raw, dict):
        return {}
    tier_model_map: Dict[int, str] = {}
    for key, value in tier_map_raw.items():
        tier = int(key)
        model_name = str(value).strip()
        if model_name:
            tier_model_map[tier] = model_name
    if not tier_model_map:
        return {}
    for tier in range(1, 6):
        if tier not in tier_model_map:
            tier_model_map[tier] = tier_model_map[max(t for t in tier_model_map if t <= tier) if any(t <= tier for t in tier_model_map) else min(tier_model_map)]
    return tier_model_map


def normalize_config(
    raw: Dict[str, Any] | None = None,
    *,
    strip_providers: bool = False,
    use_defaults: bool = True,
) -> Dict[str, Any]:
    """Unified config normalization.

    Args:
        raw: Raw config dict. None uses defaults.
        strip_providers: If True, set model_providers to {} (for inference-service).
        use_defaults: If True, fall back to build_default_runtime_config() for missing fields.
    """
    base = build_default_runtime_config() if use_defaults else {}
    raw = raw or {}

    route_order = raw.get("route_order") or base.get("route_order", list(FIVEWAY_ROUTE_ORDER))
    if route_order != list(FIVEWAY_ROUTE_ORDER):
        raise ValueError(f"route_order must exactly match {FIVEWAY_ROUTE_ORDER}")

    weights_raw = raw.get("weights", base.get("weights", dict(FIVEWAY_DEFAULT_WEIGHTS)))
    weights = _validate_weights(weights_raw, FIVEWAY_DEFAULT_WEIGHTS)

    score_bands_value = raw.get("score_bands", base.get("score_bands"))
    if not score_bands_value:
        defaults = build_default_runtime_config()
        score_bands_value = defaults["score_bands"]
        logger.warning("score_bands empty in remote config, using defaults")
    score_bands_raw_hint = str(raw.get("score_bands_raw", "")).strip()
    score_bands, score_bands_raw = _validate_score_bands(score_bands_value, score_bands_raw_hint)

    tier_map_raw = raw.get("tier_model_map", base.get("tier_model_map"))
    if not tier_map_raw:
        defaults = build_default_runtime_config()
        tier_map_raw = defaults["tier_model_map"]
        logger.warning("tier_model_map empty in remote config, using defaults")
    tier_model_map = _validate_tier_model_map(tier_map_raw)
    if not tier_model_map:
        defaults = build_default_runtime_config()
        tier_model_map = _validate_tier_model_map(defaults["tier_model_map"])

    router_alias = str(raw.get("router_alias", base.get("router_alias", DEFAULT_ROUTER_ALIAS))).strip() or DEFAULT_ROUTER_ALIAS

    # Model providers
    if strip_providers:
        model_providers: Dict[str, Dict[str, str]] = {}
    else:
        providers_raw = raw.get("model_providers", base.get("model_providers", {}))
        if not isinstance(providers_raw, dict):
            raise ValueError("model_providers must be a dict")
        model_providers = {}
        for model_name, prov in providers_raw.items():
            if not isinstance(prov, dict):
                raise ValueError(f"model_providers[{model_name}] must be a dict")
            for key in ("provider_slug", "api_key", "api_base", "upstream_model"):
                if key not in prov or not str(prov[key]).strip():
                    raise ValueError(f"model_providers[{model_name}] missing {key}")
            model_providers[str(model_name).strip()] = {
                "provider_slug": str(prov["provider_slug"]).strip(),
                "api_key": _resolve_env_vars(str(prov["api_key"]).strip()),
                "api_base": _resolve_env_vars(str(prov["api_base"]).strip()),
                "upstream_model": str(prov["upstream_model"]).strip(),
            }
            resolved_key = model_providers[str(model_name).strip()]["api_key"]
            if not resolved_key or "${" in resolved_key:
                raise ValueError(
                    f"model_providers[{model_name}].api_key resolved to empty or unresolved value"
                )

    return {
        "router_alias": router_alias,
        "route_order": list(FIVEWAY_ROUTE_ORDER),
        "weights": weights,
        "score_bands_raw": score_bands_raw,
        "score_bands": score_bands,
        "tier_model_map": tier_model_map,
        "model_providers": model_providers,
    }


# Convenience aliases for backward compatibility
def normalize_runtime_config(raw: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return normalize_config(raw, strip_providers=False, use_defaults=True)


def normalize_inference_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_config(raw, strip_providers=True, use_defaults=False)


def normalize_profile_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single profile config for inference-service.

    Only validates weights and route_order. Fills in default score_bands and
    tier_model_map so the engine's internal computation doesn't break, but
    these values are NOT exposed in the ClassifyResponse (D-08).
    """
    route_order = raw.get("route_order") or list(FIVEWAY_ROUTE_ORDER)
    if route_order != list(FIVEWAY_ROUTE_ORDER):
        raise ValueError(f"route_order must exactly match {FIVEWAY_ROUTE_ORDER}")

    weights_raw = raw.get("weights", dict(FIVEWAY_DEFAULT_WEIGHTS))
    weights = _validate_weights(weights_raw, FIVEWAY_DEFAULT_WEIGHTS)

    # Fill defaults for engine internals (score_bands/tier_model_map required by engine)
    defaults = build_default_runtime_config()
    score_bands, score_bands_raw = _validate_score_bands(defaults["score_bands"])
    tier_model_map = _validate_tier_model_map(defaults["tier_model_map"])

    return {
        "router_alias": raw.get("slug", "unknown"),
        "route_order": list(FIVEWAY_ROUTE_ORDER),
        "weights": weights,
        "score_bands_raw": score_bands_raw,
        "score_bands": score_bands,
        "tier_model_map": tier_model_map,
        "model_providers": {},
    }


def clone_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "router_alias": config["router_alias"],
        "route_order": list(config["route_order"]),
        "weights": dict(config["weights"]),
        "score_bands_raw": config["score_bands_raw"],
        "score_bands": list(config["score_bands"]),
        "tier_model_map": dict(config["tier_model_map"]),
        "model_providers": {k: dict(v) for k, v in config.get("model_providers", {}).items()},
    }
