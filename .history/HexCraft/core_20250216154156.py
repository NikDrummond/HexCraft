

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
        # make sure that q + r + s == 0
        assert coordinate.sum == 0, "q + r + s must be 0"

        self.coordinate = coordinate
    
    def q(self):
        return self.coordinate[0]
    
    def r(self):
        return self.coordinate[1]
    
    def s(self):
        return self.coordinate[2]