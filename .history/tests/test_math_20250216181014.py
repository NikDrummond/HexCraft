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
    assert jnp.array_equal(result,expected), 'Not adding Hexagon coordinates correctly'

# subtract test
def test_subtract():
    c1 = jnp.array([0,1,-1])
    c2 = jnp.array([1,0,-1])
    h1 = Hexagon(c1)
    h2 = Hexagon(c2)
    expected = c1 - c2
    result = hexMath.hex_subtract(h1,h2).coordinate
    assert jnp.array_equal(result,expected), 'Not subtracting Hexagon coordinates correctly'

# multiply test
def test_multiply():
    c = jnp.array([0,1,-1])
    k = 3
    h = Hexagon(c)
    expected = c * k
    result = hexMath.hex_multiply(h,k).coordinate
    assert jnp.array_equal(result,expected), 'Not multiplying Hexagon coordinates correctly'

# assert k is int in multiply test
def test_multiply_float_reject():
    h = Hexagon(jnp.array([0,1,-1]))
    k = 3.5
    with pytest.raises(AssertionError, match = 'Coordinate scaling factor k must be an integer'):
        hexMath.hex_multiply(h,k)

# test hex_length (axial coordinate distance from origin)
def test_length():
    h = Hexagon(jnp.array([2,-2,0]))
    assert hexMath.
