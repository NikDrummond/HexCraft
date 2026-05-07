import numpy as np
from .core import Hexagon,Hexagons

_FLAT_TOP = np.array([[3 / 2, 0], [-np.sqrt(3) / 2, -np.sqrt(3)]], dtype=float)
_POINTY_TOP = np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, -3 / 2]], dtype=float)

def _points_flat_top(a: np.ndarray, size: float = 1.0) -> np.ndarray:
    return size * (_FLAT_TOP @ a.T)

def _point_flat_top(a: np.ndarray, size: float = 1.0) -> np.ndarray:
    return size * (_FLAT_TOP @ a)

def _points_flat_side(a: np.ndarray, size: float = 1.0) -> np.ndarray:
    return size * (_POINTY_TOP @ a.T)

def _point_flat_side(a: np.ndarray, size: float = 1.0) -> np.ndarray:
    return size * (_POINTY_TOP @ a)

def hex_2D_conversion(
    a: Hexagon | Hexagons,
    center: bool = True,
    center_point: np.ndarray = np.array([0.0, 0.0]),
    method: str = "flat_top",
    size: float = 1.0
) -> np.ndarray:

    valid_methods = ["flat_top", "flat_side"]
    axial_hex_coords = a.axial_coords()

    if isinstance(a,Hexagon):
        if method == "flat_top":
            arr = _point_flat_top(axial_hex_coords, size=size)
        elif method == "flat_side":
            arr = _point_flat_side(axial_hex_coords, size=size)
        else:
            raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")
        
    if isinstance(a,Hexagons):
        if method == "flat_top":
            arr = _points_flat_top(axial_hex_coords, size=size).T
        elif method == "flat_side":
            arr = _points_flat_side(axial_hex_coords, size=size).T
        else:
            raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")
        
    if center:
        if np.array_equal(center_point, np.array([0,0])):
            arr = arr - arr.mean(axis = 0)
        else:
            arr = arr - center_point

    return arr