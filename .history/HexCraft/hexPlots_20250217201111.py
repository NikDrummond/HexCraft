import jax.numpy as jnp
from jax import jit, vmap

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc

from .core import Hexagon, Hexagons
from .hexLayouts import hex_2D_conversion



@jit
def _flat_top_verts(center:jnp.ndarray, radius:float) -> jnp.ndarray:

    angles = jnp.deg2rad(jnp.array([0,60,120,180,240,300]))
    x = center[0] + radius * jnp.cos(angles)
    y = center[1] + radius * jnp.sin(angles)
    # In this case 
    return jnp.column_stack((x,y))

@jit
def _point_top_verts(center:jnp.ndarray, radius:float) -> jnp.ndarray:

    angles = jnp.deg2rad(jnp.array([30,90,150,210,270,330]))
    x = center[0] + radius * jnp.cos(angles)
    y = center[1] + radius * jnp.sin(angles)
    return jnp.column_stack((x,y))

def hex_Patches(hexs: Hexagons, s: float = 1, method: str = 'flat_side', **kwargs) -> mc.PolyCollection:
    coords = hex_2D_conversion(hexs, s = s, method = method)
    hexagons = vmap(_point_top_verts, in_axes=(0, None))(coords, s)
    # Create a PolyCollection with individual colors
    return mc.PolyCollection(hexagons, **kwargs)

def get_ax_limits(hexs:Hexagons, padding: float = 1)
