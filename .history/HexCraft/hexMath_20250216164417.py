from .core import Hexagon
import jax.numpy as jn[]

def hex_equal(a: Hexagon, b: Hexagon) -> bool:
    return bool(jnp.array_equal(a.coordinate,b.coordinate))

def hex_add(a: Hexagon, b: Hexagon) -> Hexagon:
    return Hexagon(a.coordinate + b.coordinate)

def hex_subtract(a: Hexagon, b: Hexagon) -> Hexagon:
    return Hexagon(a.coordinate - b.coordinate)

def hex_multiply(a: Hexagon, k: int) -> Hexagon:
    assert isinstance(k, int), 'Coordinate scaling factor k must be an integer'
    return Hexagon(a.coordinate * k)
