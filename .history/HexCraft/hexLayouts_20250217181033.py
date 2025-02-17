from jax import jit
import jax.numpy as jnp
from .core import Hexagon,Hexagons
from GeoJax import center_points as cp
from GeoJax.core import _mat_mul

@jit
def _points_flat_top(a: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
    _flat_top = jnp.array([[3 / 2, 0], [jnp.sqrt(3) / 2, jnp.sqrt(3)]])
    return s * _mat_mul(_flat_top, a.T)

@jit
def _point_flat_top(a: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
    _flat_top = jnp.array([[3 / 2, 0], [jnp.sqrt(3) / 2, jnp.sqrt(3)]])
    return s * _mat_mul(_flat_top, a)

@jit
def _points_pointy_top(a: jnp.ndarray, s: float = 1.0) -> jnp.ndarray:
    _pointy_top = jnp.array([[jnp.sqrt(3), jnp.sqrt(3) / 2], [0, 3 / 2]])
    return s * _mat_mul(_pointy_top, a.T)

@jit
def _point_pointy_top(a:jnp.ndarray, s:float = 1.0) -> jnp.ndarray:
    _pointy_top = jnp.array([[jnp.sqrt(3), jnp.sqrt(3) / 2], [0, 3 / 2]])
    return s * _mat_mul(_pointy_top, a)

def hex_2D_conversion(
    a: Hexagon | Hexagons,
    center: bool = True,
    center_point: jnp.ndarray = jnp.array([0.0, 0.0]),
    method: str = "flat_top",
    s: float = 1.0,
) -> jnp.ndarray:

    valid_methods = ["flat_top", "point_top"]
    axial_hex_coords = a.axial_coords()
    
    if method == "flat_top":
        arr = _point_flat_top(axial_hex_coords, s=s)
    elif method == "point_top":
        arr = _point_pointy_top(axial_hex_coords, s=s)
    else:
        raise ValueError(f"Invalid method '{method}'. Expected one of {valid_methods}.")
    
    if center:
        if jnp.array_equal(center_point, jnp.array([0,0])):
            arr = cp(arr)
        else:
            arr = cp(arr, center_point)

    return arr