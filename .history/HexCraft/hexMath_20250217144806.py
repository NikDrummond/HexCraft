from .core import Hexagon, Hexagons
import jax.numpy as jnp
from typing import List

def hex_equal(a: Hexagon, b: Hexagon) -> bool:
    return bool(jnp.array_equal(a.coordinates,b.coordinates))

def hex_add(a: Hexagon, b: Hexagon) -> Hexagon | Hexagons:

    arr = a.coordinates + b.coordinates
    if arr.ndim == 1:
        return Hexagon(arr)
    elif arr.ndim == 2:
        return Hexagons(arr)

def hex_subtract(a: Hexagon, b: Hexagon) -> Hexagon | Hexagons:

    arr = a.coordinates - b.coordinates
    if arr.ndim == 1:
        return Hexagon(arr)
    elif arr.ndim == 2:
        return Hexagons(arr)

def hex_multiply(a: Hexagon, k: int) -> Hexagon | Hexagons:
    assert isinstance(k, int), 'Coordinate scaling factor k must be an integer'
    
    arr = a.coordinates * k
    if arr.ndim == 1:
        return Hexagon(arr)
    elif arr.ndim == 2:
        return Hexagons(arr)

def hex_length(a:Hexagon) -> int | jnp.ndarray:

    if isinstance(a, Hexagon):
        return int(abs(a.coordinates).sum()/2)
    elif isinstance(a, Hexagons):
        lens = abs(a.coordinates).sum(axis = -1)/2
        return lens.astype(int)
    

def hex_distance(a:Hexagon, b:Hexagon) -> int:
    return hex_length(hex_subtract(a,b))

def _get_direction_hex(direction:int, directions: jnp.ndarray) -> Hexagon:
    return Hexagon(directions[direction])

def hex_neighbour(a:Hexagon, direction: int | None = None, keep_a: bool = True) -> Hexagon | Hexagons:
    
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
        if keep_a:
        return Hexagons(neighbours)
    
def hex_diagonal_neighbour(a:Hexagon, direction: int | None = None) -> Hexagon | Hexagons:
    
    # specify directions - order (right top, right, right bottom, left bottom, left, left top)
    directions = jnp.array([
        [1, -2, 1], 
        [2, -1, -1], 
        [1, 1, -2], 
        [-1, 2, -1], 
        [-2, 1, 1], 
        [-1, -1, 2]
    ], dtype = int)

    if direction != None:
        b = _get_direction_hex(direction,directions)
        return hex_add(a,b)
    else:
        neighbours = a.coordinates + directions
        return Hexagons(neighbours)