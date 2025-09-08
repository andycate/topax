import jax
import jax.numpy as jnp

from cax._utils import norm, rotation_matrix

def sphere(radius, position=[0., 0., 0.]):
    position = jnp.atleast_2d(position)
    def _sphere(p):
        p = jnp.atleast_2d(p)
        return norm(p - position) - radius
    return _sphere

def box(dims, position=[0., 0., 0.]):
    s = jnp.atleast_2d(dims)
    position = jnp.atleast_2d(position)
    def _box(p):
        d = jnp.abs(p-position)-s
        return jnp.linalg.norm(jnp.maximum(d,0.0), axis=-1) + jnp.minimum(jnp.max(d, axis=-1),0.0)
    return _box

def cylinder(radius, height):
    radh = jnp.array([[radius, height]])
    def _cylinder(p):
        d = jnp.abs(jnp.c_[jnp.linalg.norm(p[:,:2], axis=-1),p[:,2]]) - radh
        return jnp.minimum(jnp.max(d, axis=-1),0.0) + jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
    return _cylinder

# def capped_cylinder(a, b, r):
#     a = jnp.atleast_2d(a)
#     b = jnp.atleast_2d(b)
#     ba = b - a
#     baba = (ba * ba).sum(axis=-1)
#     def _capped_cylinder(p):
#         pa = p - a
#         paba = (pa * ba).sum(axis=-1)
#         x = jnp.linalg.norm(pa*baba-ba*paba, axis=-1) - r*baba
#         y = jnp.abs(paba-baba*0.5)-baba*0.5
#         x2 = x*x
#         y2 = y*y*baba
#         # d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
#         # return sign(d)*sqrt(abs(d))/baba
#     return _capped_cylinder

def translate(f, d):
    d = jnp.atleast_2d(d)
    def _translate(p):
        return f(p - d)
    return _translate

def scale(f, factor):
    def _scale(p):
        return f(p/factor) * factor
    return _scale

def union(*args):
    def _union(p):
        distances = jnp.zeros((p.shape[0], len(args)))
        for i, f in enumerate(args):
            distances = distances.at[:, i].set(f(p))
        return jnp.min(distances, axis=-1)
    return _union

def intersect(*args):
    def _intersect(p):
        distances = jnp.zeros((p.shape[0], len(args)))
        for i, f in enumerate(args):
            distances = distances.at[:, i].set(f(p))
        return jnp.max(distances, axis=-1)
    return _intersect

def subtract(f, tool):
    def _subtract(p):
        return jnp.maximum(-tool(p), f(p))
    return _subtract

# def circular_pattern(f, cnt):
#     rotations = jnp.zeros((cnt, 3, 3))
#     for i in range(cnt):
#         rotations = rotations.at[i].set(rotation_matrix(-i * 2 * jnp.pi / cnt, 'y'))
#     def _circular_pattern(p):
#         p_mod = rotations @ p
#         return jnp.min(jax.vmap(f)(p_mod))
#     return _circular_pattern

def test_sdf():
    part1 = union(
        sphere(0.5),
        box([0.3, 0.3, 0.3], position=[0.5, 0.5, 0.5]),
    )
    return part1
