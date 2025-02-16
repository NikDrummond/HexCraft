import jax.numpy as jnp
from typing import List

def _coordinate_int_conversion(arr: jnp.ndarray) -> jnp.ndarray:

    # if array is already integers we can return it
    if jnp.issubdtype(arr.dtype, jnp.integer):
        return arr
    elif jnp.all(arr == jnp.round(arr)):
        return arr.astype(int)
    else:
        raise TypeError('Input coordinates must be integer values')

class Hexagon:

    def __init__(self, coordinate: jnp.ndarray):
        
        # make sure we are given integers
        coordinate = _coordinate_int_conversion(coordinate)
        # make sure that the input is length 3
        assert len(coordinate) == 3, "Input coordinate must be 3D for q, r, and s."
        # make sure that q + r + s == 0
        assert coordinate.sum() == 0, "q + r + s must be 0"

        self.coordinate = coordinate
    
    def q(self):
        return self.coordinate[0]
    
    def r(self):
        return self.coordinate[1]
    
    def s(self):
        return self.coordinate[2]
    
class Hexagons:

    def __init__(self, coordinates: jnp.ndarray):

        coordinates = _coordinate_int_conversion(coordinates)
        assert coordinates.shape[1] == 3, "Input coordinates must be 3D for q, r, and s."
        assert coordinates.sum() == 0, "q + r + s must be 0 for all coordinates"
        self.coordinates = coordinates

    def get_hexagon(self, i:int) -> Hexagon:
        return Hexagon(self.coordinates[i])
    
    def hexagons_list(self) -> List:
        return [Hexagon[i] for i in self.coordinates]
    
    def all_q(self) -> jnp.ndarray:
        return self.coordinates[:,0]
    
    def all_r(self) -> jnp.ndarray:
        return self.coordinates[:,1]
    
    def all_s(self) -> jnp.ndarray:
        return self.coordinates[:,2]