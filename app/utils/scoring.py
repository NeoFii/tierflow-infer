"""Score normalization and weighted computation."""

from __future__ import annotations

import math
from typing import Dict, Tuple

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


def scale_final_score_to_0_10(
    score_raw: float,
    lower: float = 0.40,
    upper: float = 1.45,
) -> float:
    if upper <= lower:
        return 5.0
    score_0_10 = 10.0 * (float(score_raw) - lower) / (upper - lower)
    return float(max(0.0, min(10.0, score_0_10)))


def compute_weighted_total_score_0_10(
    route_scores_0_2: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Aggregate 5-way route scores with equal weighting (1.0 each).

    Returns (total_score_0_10, per-route components used for the average).
    """
    components = {
        name: float(route_scores_0_2[name])
        for name in FIVEWAY_ROUTE_ORDER
    }
    average_0_2 = sum(components.values()) / len(FIVEWAY_ROUTE_ORDER)
    total_score_0_10 = average_0_2 * 5.0
    return float(total_score_0_10), components


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
