from jax import jit
import jax.numpy as jnp
from .core import Hexagon,Hexagons

@jit
def _mat_mul(a:jnp.ndarray,b:jnp.ndarray) -> jnp.ndarray:
    return a @ b

@jit
def _point_flat_top(a:jnp.ndarray) -> jnp.ndarray:
    _flat_top = jnp.array([[3/2,0],[jnp.sqrt(3)/2,jnp.sqrt(3)]])
    return _mat_mul(a,_flat_top)

@jit
def _point_pointy_top(a:jnp.ndarray) -> jnp.ndarray:
    _pointy_top =jnp.array([[jnp.sqrt(3),jnp.sqrt(3)/2],[0,3/2]])
    return _mat_mul(a,_pointy_top)

def hex_2D_conversion(a: Hexagon | Hexagons, method = 'flat_top') -> jnp.ndarray:

    assert method in ['flat_top','point_top'], 'Method must be flat_top or point_top'
    axial_hex_coords = a.axial_coords()
    if method == 'flat_top':
        return _point_flat_top(axial_hex_coords)
    elif method == 'point_top':
        return _point_pointy_top(axial_hex_coords)
    else:
        raise ValueErrorf"Invalid method '{method}'. Expected one of {valid_methods}.")

