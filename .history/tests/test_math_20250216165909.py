import pytest
from HexCraft import hexMath
from HexCraft.core import Hexagon
import jax.numpy as jnp

# equality correct test
def test_correct_equality():
    a = Hexagon(jnp.array([0,1,-1]))
    assert hexMath.hex_equal(a,a), 'Not correctly accepting the same Hexagon'
# equality fail test
def test_wrong_equality():
    a = Hexagon(jnp.array([0,1,-1]))
    b = Hexagon(jnp.array([1,0,-1]))
    assert not hexMath.hex_equal(a,b), 'Not correctly rejecting different Hexagons'

# add test
def test_add():
    c1 = jnp.array([0,1,-1])
    c2 = jnp.array([1,0,-1])
    h1 = Hexagon(c1)
    h2 = Hexagon(c2)
    expected = c1 + c2
    result = hexMath.hex_add(h1,h2).coordinate
    assert jnp.array_equal(result,expected), 'Not adding Hexagons correctly'

# subtract test
def test_subtract():
    c1 = jnp.array([0,1,-1])
    c2 = jnp.array([1,0,-1])
    h1 = Hexagon(c1)
    h2 = Hexagon(c2)
    expected = c1 - c2
    result = hexMath.hex_subtract(h1,h2).coordinate
    assert jnp.array_equal(result,expected), 'Not adding Hexagons correctly'

# multiply test

# assert k is int in multiply test