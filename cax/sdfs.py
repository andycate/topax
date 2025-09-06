import jax
import jax.numpy as jnp

from cax._utils import norm

def sphere(radius, position=[0., 0., 0.]):
    position = jnp.atleast_1d(position)
    def _sphere(p):
        return norm(p - position) - radius
    return _sphere

def box(dims, position=[0., 0., 0.]):
    s = jnp.atleast_1d(dims)
    position = jnp.atleast_1d(position)
    def _box(p):
        d = jnp.abs(p-position)-s
        return jnp.linalg.norm(jnp.maximum(d,0.0)) + jnp.minimum(jnp.max(d),0.0)
    return _box

def translate(f, d):
    d = jnp.atleast_1d(d)
    def _translate(p):
        return f(p - d)
    return _translate

def scale(f, factor):
    def _scale(p):
        return f(p/factor) * factor
    return _scale

def union(*args):
    def _union(p):
        distances = jnp.zeros(len(args))
        for i, f in enumerate(args):
            distances = distances.at[i].set(f(p))
        return jnp.min(distances)
    return _union

def intersect(*args):
    def _intersect(p):
        distances = jnp.zeros(len(args))
        for i, f in enumerate(args):
            distances = distances.at[i].set(f(p))
        return jnp.max(distances)
    return _intersect

def subtract(f, tool):
    def _subtract(p):
        return jnp.maximum(-tool(p), f(p))
    return _subtract

def test_sdf():
    part1 = union(
        sphere(0.5),
        box([0.3, 0.3, 0.3], position=[0.5, 0.5, 0.5]),
    )
    return part1
