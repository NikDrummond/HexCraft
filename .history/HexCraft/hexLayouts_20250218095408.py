from jax import jit
import jax.numpy as jnp
from .core import Hexagon,Hexagons
from GeoJax import center_points as cp
from GeoJax.core import _mat_mul

@jit
def _points_flat_top(a: jnp.ndarray, size: float = 1.0, scale: float = 1.0) -> jnp.ndarray:
    _flat_top = jnp.array([[3 / 2, 0], [jnp.sqrt(3) / 2, jnp.sqrt(3)]])
    s = size * scale
    return s * _mat_mul(_flat_top, a.T)

@jit
def _point_flat_top(a: jnp.ndarray, size: float = 1.0, scale: float = 1.0) -> jnp.ndarray:
    _flat_top = jnp.array([[3 / 2, 0], [jnp.sqrt(3) / 2, jnp.sqrt(3)]])
    s = size * scale
    return s * _mat_mul(_flat_top, a)

@jit
def _points_flat_side(a: jnp.ndarray, size: float = 1.0, scale: float = 1.0) -> jnp.ndarray:
    _pointy_top = jnp.array([[jnp.sqrt(3), jnp.sqrt(3) / 2], [0, 3 / 2]])
    s = size * scale
    return s * _mat_mul(_pointy_top, a.T)

@jit
def _point_flat_side(a:jnp.ndarray, size:float = 1.0, scale: float = 1.0) -> jnp.ndarray:
    _pointy_top = jnp.array([[jnp.sqrt(3), jnp.sqrt(3) / 2], [0, 3 / 2]])
    s = size * scale
    return s * _mat_mul(_pointy_top, a)

def hex_2D_conversion(
    a: Hexagon | Hexagons,
    center: bool = True,
    center_point: jnp.ndarray = jnp.array([0.0, 0.0]),
    method: str = "flat_top",
    size: float = 1.0,
    scale: float = 1.0
) -> jnp.ndarray:

    valid_methods = ["flat_top", "flat_side"]
    axial_hex_coords = a.axial_coords()

    if isinstance(a,Hexagon):
        if method == "flat_top":
            arr = _point_flat_top(axial_hex_coords, size=size, scale = scale)
        elif method == "flat_side":
            arr = _point_flat_side(axial_hex_coords, size=size, scale = scale)
        else:
            raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")
        
    if isinstance(a,Hexagons):
        if method == "flat_top":
            arr = _points_flat_top(axial_hex_coords, size=s).T
        elif method == "flat_side":
            arr = _points_flat_side(axial_hex_coords, s=s).T
        else:
            raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")
        
    if center:
        if jnp.array_equal(center_point, jnp.array([0,0])):
            arr = cp(arr)
        else:
            arr = cp(arr, center_point)

    return arr