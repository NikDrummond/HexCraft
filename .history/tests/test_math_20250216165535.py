import pytest
from HexCraft import hexMath
from HexCraft.core import Hexagon
import jax.numpy as jnp

# equality correct test
def test_correct_equality():
    a = Hexagon(jnp.array([0,1,-1]))
    assert hexMath.hex_equal(a,a)
# equality fail test
def test_wrong_equality():
    a = Hexagon(jnp.array([0,1,-1]))
    b = Hexagon(jnp.array([1,0,-1]))
    assert hexMath.hex_equal(a,a)
# add test

# subtract test

# multiply test

# assert k is int in multiply test