import jax.numpy as jnp
from jax import jit, vmap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc

from .core import Hexagon, Hexagons
from .hexLayouts import hex_2D_conversion


# @jit
# def _flat_top_verts(center: jnp.ndarray, size: float, scale: float) -> jnp.ndarray:

#     angles = jnp.deg2rad(jnp.array([0, 60, 120, 180, 240, 300]))
#     s = size * scale
#     x = center[0] + s * jnp.cos(angles)
#     y = center[1] + s * jnp.sin(angles)
#     # In this case
#     return jnp.column_stack((x, y))

@jit
def _flat_top_verts(center: jnp.ndarray, size: float, scale: float) -> jnp.ndarray:
    
    angles = jnp.deg2rad(jnp.array([0, 60, 120, 180, 240, 300]))
    # Compute unit offsets (for size=1)
    offsets = jnp.column_stack((jnp.cos(angles), jnp.sin(angles)))
    # Multiply the unit offsets by the size to get the unscaled offsets,
    # then scale them further by 'scale'
    return center + scale * (size * offsets)



@jit
def _point_top_verts(center: jnp.ndarray, size: float, scale: float) -> jnp.ndarray:

    angles = jnp.deg2rad(jnp.array([30, 90, 150, 210, 270, 330]))
    s = size * scale
    x = center[0] + s * jnp.cos(angles)
    y = center[1] + s * jnp.sin(angles)
    return jnp.column_stack((x, y))


def hex_Patches(
    hexs: Hexagons, size: float = 1.0, scale: float = 1.0, method: str = "flat_side", **kwargs
) -> mc.PolyCollection:
    coords = hex_2D_conversion(hexs, size=size, method=method)
    if method == 'flat_top':
        hexagons = vmap(_flat_top_verts, in_axes=(0, None, None))(coords, size, scale)
    elif method == 'flat_side':
        hexagons = vmap(_point_top_verts, in_axes=(0, None, None))(coords, size, scale)
    # Create a PolyCollection with individual colors
    return mc.PolyCollection(hexagons, **kwargs)


def get_ax_limits(hexs: Hexagons, size: float = 1.0, scale: float = 1.0, padding: float = 1.0, method: str = 'flat_side') -> jnp.ndarray:
    coords = hex_2D_conversion(hexs, size=size, method=method)
    arr = jnp.array(
        [
            [coords[:, 0].min() - padding, coords[:, 0].max() + padding],
            [coords[:, 1].min() - padding, coords[:, 1].max() + padding],
        ]
    )
    return arr
