#!/usr/bin/env python3
"""Shared box trust-region helpers for Taylor-linearized CBF filters."""

from __future__ import annotations

import numpy as np


def box_trust_region_bounds(
    u_nom: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    radius: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect box input bounds with |u - u_nom|_inf <= radius."""
    u_nom = np.asarray(u_nom, dtype=float)
    u_min = np.asarray(u_min, dtype=float)
    u_max = np.asarray(u_max, dtype=float)
    if radius is None:
        return u_min.copy(), u_max.copy()
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"trust-region radius must be finite and positive, got {radius}")
    return np.maximum(u_min, u_nom - radius), np.minimum(u_max, u_nom + radius)


def box_trust_region_metrics(
    u: np.ndarray,
    u_nom: np.ndarray,
    radius: float | None,
    *,
    relative_tolerance: float = 1e-6,
) -> dict[str, float | bool]:
    """Return displacement and boundary-activity diagnostics."""
    delta_linf = float(
        np.max(np.abs(np.asarray(u, dtype=float) - np.asarray(u_nom, dtype=float)))
    )
    if radius is None:
        return {
            "u_delta_linf": delta_linf,
            "trust_region_ratio": 0.0,
            "trust_region_active": False,
        }
    ratio = delta_linf / float(radius)
    return {
        "u_delta_linf": delta_linf,
        "trust_region_ratio": float(ratio),
        "trust_region_active": bool(ratio >= 1.0 - float(relative_tolerance)),
    }
