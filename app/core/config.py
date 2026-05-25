"""Global configuration: ML constants, model path loading, service settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from pydantic import model_validator

from app.common.config import BaseServiceSettings


# ---------------------------------------------------------------------------
# Five-way route order (fixed, must match training)
# ---------------------------------------------------------------------------
FIVEWAY_ROUTE_ORDER: List[str] = ["纠错", "工具调用", "通用任务", "任务拆解", "编程"]
FIVEWAY_DEFAULT_WEIGHTS: Dict[str, float] = {
    "纠错": 1.0,
    "工具调用": 1.0,
    "通用任务": 1.0,
    "任务拆解": 1.0,
    "编程": 1.0,
}

ROUTE_ERROR = FIVEWAY_ROUTE_ORDER[0]
ROUTE_TOOL = FIVEWAY_ROUTE_ORDER[1]
ROUTE_GENERAL = FIVEWAY_ROUTE_ORDER[2]
ROUTE_TASK = FIVEWAY_ROUTE_ORDER[3]
ROUTE_CODE = FIVEWAY_ROUTE_ORDER[4]

# ---------------------------------------------------------------------------
# 0-2 normalization ranges (from training calibration)
# ---------------------------------------------------------------------------
NORMALIZE_RANGES: Dict[str, Tuple[float, float]] = {
    "纠错": (0.46160000562667847, 1.7732000350952148),
    "工具调用": (0.31450000405311584, 2.156100034713745),
    "通用任务": (-0.13300000131130219, 1.2163000106811523),
    "任务拆解": (0.008700000122189522, 1.9020999670028687),
    "编程": (0.13930000364780426, 0.7321000099182129),
}

# ---------------------------------------------------------------------------
# Proto relevance weighting
# ---------------------------------------------------------------------------
PROTO_LABEL_ORDER: List[str] = ["swe", "tool", "gaia", "task", "prog"]
PROTO_ROUTE_TO_LABEL: Dict[str, str] = {
    "纠错": "swe",
    "工具调用": "tool",
    "通用任务": "gaia",
    "任务拆解": "task",
    "编程": "prog",
}
PROTO_LABEL_TO_ROUTE: Dict[str, str] = {v: k for k, v in PROTO_ROUTE_TO_LABEL.items()}
FINAL_SCORE_LOWER: float = 0.40
FINAL_SCORE_UPPER: float = 1.45
FINAL_SCORE_SOURCE: str = "proto_weighted_0_2"


# ---------------------------------------------------------------------------
# Model paths loader
# ---------------------------------------------------------------------------
class ModelPathsConfig:
    """Loads model file paths from a dict or JSON file."""

    DEFAULT_HOOK_TARGET_TEMPLATE = "model.layers.{layer}.self_attn.o_proj"

    def __init__(self, raw: Dict[str, Any]):
        self.qwen_backbone: str = raw["qwen_backbone"]
        self.device: str = raw.get("device", "cuda:0")
        self.max_input_length: int = raw.get("max_input_length", 4096)
        self.proto_artifact: str = raw.get("proto_artifact", "")
        self._hook_target_template: str = raw.get(
            "hook_target_template", self.DEFAULT_HOOK_TARGET_TEMPLATE
        )

        routers = raw.get("routers", {})
        self.routers: Dict[str, Dict[str, Any]] = {}
        for name, cfg in routers.items():
            entry: Dict[str, Any] = {
                "heads": [tuple(h) for h in cfg["heads"]],
                "model": cfg["model"],
                "scaler": cfg["scaler"],
            }
            if "meta" in cfg:
                entry["meta"] = cfg["meta"]
            self.routers[name] = entry

    @classmethod
    def from_file(cls, config_path: str) -> "ModelPathsConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)
        return cls(raw)

    def get_heads(self, name: str) -> List[Tuple[int, int]]:
        return self.routers[name]["heads"]

    def get_model_path(self, name: str) -> str:
        return self.routers[name]["model"]

    def get_scaler_path(self, name: str) -> str:
        return self.routers[name]["scaler"]

    def get_meta_path(self, name: str) -> str | None:
        return self.routers[name].get("meta")

    def get_hook_target(self, base_model: Any, layer_idx: int) -> Any:
        path = self._hook_target_template.replace("{layer}", str(layer_idx))
        obj = base_model
        for attr in path.split("."):
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj


def load_model_paths(config_path: str | None = None) -> ModelPathsConfig:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_paths.json")
        config_path = os.path.abspath(config_path)
    return ModelPathsConfig.from_file(config_path)


# ---------------------------------------------------------------------------
# Service settings (pydantic-settings, extends BaseServiceSettings)
# ---------------------------------------------------------------------------
class InferenceSettings(BaseServiceSettings):
    """Inference-service specific settings."""

    SERVICE_NAME: str = "inference-service"
    PORT: int = 8001

    INFERENCE_HOST: str = "0.0.0.0"
    INFERENCE_SERVICE_SECRET: str = ""
    INFERENCE_ALLOW_INSECURE_DEV: bool = False

    ROUTER_MODEL_PATHS: str = ""

    API_SERVICE_URL: str = "http://127.0.0.1:8000"

    # DEPRECATED: Use API_SERVICE_URL. Kept for backward compat with existing deploy scripts.
    ADMIN_SERVICE_URL: str = "http://127.0.0.1:8001"
    CONFIG_REFRESH_INTERVAL_SECONDS: int = 60
    CONFIG_FETCH_TIMEOUT_SECONDS: float = 5.0

    GPU_CONCURRENCY_LIMIT: int = 8

    # Override base class validator: inference-service has no DB/JWT/Redis
    @model_validator(mode="after")
    def validate_required_fields(self) -> "InferenceSettings":
        if not self.INTERNAL_SECRET:
            raise ValueError("INTERNAL_SECRET must be configured")
        if len(self.INTERNAL_SECRET) < 32:
            raise ValueError("INTERNAL_SECRET length must be at least 32")
        return self


@lru_cache
def get_settings() -> InferenceSettings:
    return InferenceSettings()
