"""Depth-bin layout shared by the LSS frustum and the depth labels.

"linear" (original): K points evenly spaced over [d_min, d_max]; label edges evenly spaced (K+1).
"log": K points geometrically spaced; label edges geometrically spaced, so bins widen with distance.
The frustum points and the label edges must use the same spacing, otherwise label k means a
different depth than frustum point k.
"""
import numpy as np


def depth_bin_points(d_min: float, d_max: float, K: int, spacing: str = "linear") -> np.ndarray:
    if spacing == "log":
        return np.geomspace(d_min, d_max, K)
    return np.linspace(d_min, d_max, K)


def depth_bin_edges(d_min: float, d_max: float, K: int, spacing: str = "linear") -> np.ndarray:
    if spacing == "log":
        return np.geomspace(d_min, d_max, K + 1)
    return np.linspace(d_min, d_max, K + 1)
