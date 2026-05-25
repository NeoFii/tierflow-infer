"""Score normalization, band parsing, tier resolution, weighted computation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from app.core.config import FIVEWAY_ROUTE_ORDER, NORMALIZE_RANGES


def minmax_scale_to_0_2(score: float, min_val: float, max_val: float) -> float:
    if not math.isfinite(score):
        return 1.0
    if max_val <= min_val:
        return 1.0
    clipped = max(min_val, min(max_val, float(score)))
    return 2.0 * (clipped - min_val) / (max_val - min_val)


def norm_0_2_to_bucket(score: float) -> str:
    if score < 2.0 / 3.0:
        return "level1"
    elif score < 4.0 / 3.0:
        return "level2"
    return "level3"


def raw_score_to_bucket(score: float) -> str:
    if score < 0.5:
        return "level1"
    elif score < 1.5:
        return "level2"
    return "level3"


def route_suggestion(level: str) -> str:
    return {
        "level1": "small_or_fast_model",
        "level2": "mid_model",
        "level3": "strong_model",
    }[level]


def scale_final_score_to_0_10(
    score_raw: float,
    lower: float = 0.40,
    upper: float = 1.45,
) -> float:
    if upper <= lower:
        return 5.0
    score_0_10 = 10.0 * (float(score_raw) - lower) / (upper - lower)
    return float(max(0.0, min(10.0, score_0_10)))


def level_from_0_10(score_0_10: float) -> str:
    if score_0_10 < 10.0 / 3.0:
        return "level1"
    if score_0_10 < 20.0 / 3.0:
        return "level2"
    return "level3"


def parse_score_bands(raw: str) -> List[Tuple[float, float, int]]:
    bands: List[Tuple[float, float, int]] = []
    for item in raw.split(","):
        left, _, right = item.partition(":")
        if not left or not right:
            continue
        tier = int(right.strip())
        if "-" in left:
            start_raw, _, end_raw = left.partition("-")
            start = float(start_raw.strip())
            end = float(end_raw.strip())
        else:
            start = end = float(left.strip())
        if start > end:
            raise ValueError("score band start must be <= end")
        bands.append((start, end, tier))
    if not bands:
        raise ValueError("score bands must not be empty")
    return bands


def resolve_score_band(score: float, bands: List[Tuple[float, float, int]]) -> int:
    for start, end, tier in bands:
        if start <= score <= end:
            return tier
    if score < bands[0][0]:
        return bands[0][2]
    return bands[-1][2]


def compute_weighted_total_score_0_10(
    route_scores_0_2: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    total_weight = sum(weights[name] for name in FIVEWAY_ROUTE_ORDER)
    if total_weight <= 0:
        raise ValueError("weights sum must be greater than 0")
    weighted_components = {
        name: float(route_scores_0_2[name]) * float(weights[name])
        for name in FIVEWAY_ROUTE_ORDER
    }
    weighted_average_0_2 = sum(weighted_components.values()) / total_weight
    total_score_0_10 = weighted_average_0_2 * 5.0
    return float(total_score_0_10), weighted_components


def normalize_route(route_name: str, raw_score: float) -> Tuple[float, str]:
    min_v, max_v = NORMALIZE_RANGES[route_name]
    score_0_2 = minmax_scale_to_0_2(raw_score, min_v, max_v)
    level = norm_0_2_to_bucket(score_0_2)
    return float(score_0_2), level


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + 1e-12)
