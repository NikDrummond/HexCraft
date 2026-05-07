import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc

from .core import Hexagon, Hexagons
from .hexLayouts import hex_2D_conversion

def _flat_top_verts(center: np.ndarray, size: float, scale: float) -> np.ndarray:
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    s = size * scale
    x = center[0] + s * np.cos(angles)
    y = center[1] + s * np.sin(angles)
    return np.column_stack((x, y))


def _point_top_verts(center: np.ndarray, size: float, scale: float) -> np.ndarray:
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    s = size * scale
    x = center[0] + s * np.cos(angles)
    y = center[1] + s * np.sin(angles)
    return np.column_stack((x, y))


def hex_Patches(
    hexs: Hexagons, size: float = 1.0, scale: float = 1.0, method: str = "flat_side", **kwargs
) -> mc.PolyCollection:
    coords = hex_2D_conversion(hexs, size=size, method=method)
    if method == 'flat_top':
        hexagons = np.stack([_flat_top_verts(c, size, scale) for c in coords], axis=0)
    elif method == 'flat_side':
        hexagons = np.stack([_point_top_verts(c, size, scale) for c in coords], axis=0)
    # Create a PolyCollection with individual colors
    return mc.PolyCollection(hexagons, **kwargs)


def get_ax_limits(hexs: Hexagons, size: float = 1.0, padding: float = 1.0, method: str = 'flat_side') -> np.ndarray:
    coords = hex_2D_conversion(hexs, size=size, method=method)
    arr = np.array(
        [
            [coords[:, 0].min() - padding, coords[:, 0].max() + padding],
            [coords[:, 1].min() - padding, coords[:, 1].max() + padding],
        ]
    )
    return arr
