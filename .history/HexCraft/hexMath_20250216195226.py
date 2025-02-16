from .core import Hexagon, Hexagons
import jax.numpy as jnp
from typing import List

def hex_equal(a: Hexagon, b: Hexagon) -> bool:
    return bool(jnp.array_equal(a.coordinates,b.coordinates))

def hex_add(a: Hexagon, b: Hexagon) -> Hexagon:
    arr = 
    return Hexagon(a.coordinates + b.coordinates)

def hex_subtract(a: Hexagon, b: Hexagon) -> Hexagon:
    return Hexagon(a.coordinates - b.coordinates)

def hex_multiply(a: Hexagon, k: int) -> Hexagon:
    assert isinstance(k, int), 'Coordinate scaling factor k must be an integer'
    return Hexagon(a.coordinates * k)

def hex_length(a:Hexagon) -> int | jnp.ndarray:

    if isinstance(a, Hexagon):
        return int(abs(a.coordinates).sum()/2)
    elif isinstance(a, Hexagons):
        lens = abs(a_neigh.coordinates).sum(axis = -1)/2
        return lens.astype(int)
    

def hex_distance(a:Hexagon, b:Hexagon) -> int:
    return hex_length(hex_subtract(a,b))

def _get_direction_hex(direction:int, directions: jnp.ndarray) -> Hexagon:
    return Hexagon(directions[direction])

def hex_neighbour(a:Hexagon, direction: int | None = None) -> Hexagon | List:
    
    # specify directions - order (top, right top, right bottom, bottom, left bottom, left top)
    directions = jnp.array([
        [0, -1, 1], 
        [1, -1, 0], 
        [1, 0, -1], 
        [0, 1, -1], 
        [-1, 1, 0], 
        [-1, 0, 1]
    ], dtype = int)

    if direction != None:
        b = _get_direction_hex(direction,directions)
        return hex_add(a,b)
    else:
        neighbours = a.coordinates + directions
        return Hexagons(neighbours)