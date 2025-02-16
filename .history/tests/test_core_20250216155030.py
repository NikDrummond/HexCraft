import pytest
from HexCraft import core
import numpy as np
import jax.numpy as jnp


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
def test_int_return_jax():
    input_array = np.array([1,2,-3])
        input_array = np.array([1,2,-3])

    result = core._coordinate_int_conversion(input_array)
    assert jnp.array_equal(input_array, result), 'Numpy integer input not preserved'


# test returns integers if floats that can be converted are given - numpy


# test raises error if cannot be converted to floats - jax


# test raises error if cannot be converted to floats - numpy


