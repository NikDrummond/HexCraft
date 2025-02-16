import pytest
from HexCraft import core
import numpy as np
import jax.numpy as jnp
import re


### test integer conversion

# test returns integers if integers given - jax
def test_int_return_jax():
    input_array = jnp.array([1,2,-3])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(input_array, result), 'Jax integer input not preserved'

# test returns integers if integers given - numpy
def test_int_return_jax():
    input_array = np.array([1,2,-3])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(input_array, result), 'Numpy integer input not preserved'

# test returns integers if floats that can be converted are given - jax
def test_int_conversion_jax():
    expected = jnp.array([1.0,2.0,-3.0])
    input_array = jnp.array([1,2,-3])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(expected, result), 'Jax float input not properly converted'

# test returns integers if floats that can be converted are given - numpy
def test_int_conversion_np():
    expected = np.array([1.0,2.0,-3.0])
    input_array = jnp.array([1,2,-3])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(expected, result), 'Numpy float input not properly converted'

# test raises error if cannot be converted to floats - jax
def test_int_conversion_TypeError_jax():
    input_array = jnp.array([1.0,2.5,3.2])
    with pytest.raises(TypeError, match = 'Input coordinates must be integer values'):
        core._coordinate_int_conversion(input_array)

# test raises error if cannot be converted to floats - Numpy
def test_int_conversion_TypeError_jax():
    input_array = np.array([1.0,2.5,3.2])
    with pytest.raises(TypeError, match = 'Input coordinates must be integer values'):
        core._coordinate_int_conversion(input_array)

# Repeat the above tests but for 2d data (only for jax)

# test returns integers if integers given - 2d
def test_int_return_2d():
    input_array = jnp.array([[1,2,-3],[1,2,-3]])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(input_array, result), '2D integer input not preserved'

# test returns integers if floats that can be converted are given - jax
def test_int_conversion_2d():
    expected = jnp.array([[1.0,2.0,-3.0],[1.0,2.0,-3.0]])
    input_array = jnp.array([[1,2,-3],[1,2,-3]])
    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(expected, result), '2D float input not properly converted'

# test raises error if cannot be converted to floats - jax
def test_int_conversion_TypeError_2d():
    input_array = jnp.array([[1.0,2.5,3.2],[1.0,2.5,3.2]])
    with pytest.raises(TypeError, match = 'Input coordinates must be integer values'):
        core._coordinate_int_conversion(input_array)

### Hexagon tests

# Given an integer coordinate array, make sure the coordinates of output are the same
def test_Hexagon_keeps_coords():
    input_array = jnp.array([0,1,-1])
    result = core.Hexagon(input_array).coordinates
    assert jnp.array_equal(input_array, result)

def test_Hexagon_qrs():
    input_array = jnp.array([0,1,-1])
    Hex = core.Hexagon(input_array)

    assert Hex.q() == input_array[0], 'q note returned properly'
    assert Hex.r() == input_array[1], 'r note returned properly'
    assert Hex.s() == input_array[2], 's note returned properly'

# make sure we raise the proper error when more than 3 values are given
def test_Hexagon_coord_len_error():
    input_array = jnp.array([0,1,3,-4])
    with pytest.raises(AssertionError, match = "Input coordinate must be 3D for q, r, and s."):
        core.Hexagon(input_array)

# make sure we raise the proper error when sum of input is not 0
def test_Hexagon_coord_sum_error():
    input_array = jnp.array([0,1,-3])
    with pytest.raises(AssertionError, match = re.escape("q + r + s must be 0")):
        core.Hexagon(input_array)

### Hexagons tests

# Given an integer coordinate array, make sure the coordinates of output are the same
def test_Hexagon_keeps_coords():
    input_array = jnp.array([[0,1,-1],[1,0,-1]])
    result = core.Hexagon(input_array).coordinates
    assert jnp.array_equal(input_array, result)

