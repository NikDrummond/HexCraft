import pytest
from HexCraft import core
import numpy as np
import jax.numpy as jnp


### test integer conversion

# test returns integers if integers given - jax
def test_int_return_jax():
    input_array = jnp.array([1,2,-3])
    result = core._
# test returns integers if integers given - numpy

# test returns integers if floats that can be converted are given - jax

# test returns integers if floats that can be converted are given - numpy

# test raises error if cannot be converted to floats - jax

# test raises error if cannot be converted to floats - numpy


