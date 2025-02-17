from jax import jit
import jax.numpy as jnp

@jit
def _mat_mul(a:jnp.ndarray,b:jnp.ndarray) -> jnp.ndarray:
    return a @ b

@jit
def _point_flat_top(a:jnp.ndarray) -> jnp.ndarray:
    _flat_top = jnp.array([[3/2,0],[jnp.sqrt(3)/2,jnp.sqrt(3)]])
    return _mat_mul(a,_flat_top)

def _point_poi
