from jax import jit
import jax.numpy as jnp

@jit
def _mat_mul(a:jnp.ndarray,b:jnp.ndarray) -> jnp.ndarray:
    return a @ b

@jit
def _point_flat_top(a)