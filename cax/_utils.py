import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

def norm(v, axis=-1, keepdims=False, eps=0.0):
    return jnp.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

def normalize(v, axis=-1, eps=1e-20):
    return v/norm(v, axis, keepdims=True, eps=eps)

def rotation_matrix(angle, axis):
    s, c = np.sin(angle), np.cos(angle)
    match axis:
        case 'x':
            return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
        case 'y':
            return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        case 'z':
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        case _:
            raise ValueError(f"Axis must be 'x' 'y' or 'z', not {axis}")
        
def rotation_matrix_about_vector(angle, axis_vec):
    axis_vec = np.asarray(axis_vec, dtype=float)
    axis_vec = axis_vec / np.linalg.norm(axis_vec)
    x, y, z = axis_vec
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])

def raycast(sdf, p0, step_n, dir):
    def f(_, p):
        return p+sdf(p)*dir
    return jax.lax.fori_loop(0, step_n, f, p0)

def camera_rays_perspective(forward, world_up, looking_at, view_size, fx=0.6):
    right = jnp.cross(forward, world_up)
    down = jnp.cross(right, forward)
    R = normalize(jnp.vstack([right, down, forward]))
    w, h = view_size
    fy = fx/w*h
    y, x = jnp.mgrid[-1.:1.:h*1j, -1.:1.:w*1j].reshape(2, -1)
    y *= fy
    x *= fx
    return normalize(jnp.c_[x, y, jnp.ones_like(x)]) @ R

def camera_rays_ortho(forward, world_up, looking_at, view_size, fx=0.6):
    right = jnp.cross(forward, world_up)
    down = jnp.cross(right, forward)
    R = normalize(jnp.vstack([right, down, forward]))
    w, h = view_size
    fy = fx/w*h
    y, x = jnp.mgrid[-1.:1.:h*1j, -1.:1.:w*1j].reshape(2, -1)
    y *= fy
    x *= fx
    return normalize(jnp.c_[x, y, jnp.ones_like(x)]) @ R

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def redraw(sdf, width, height, step_n, pos0, world_up, looking_at, fx, color):
    ray_dir = camera_rays_perspective(-pos0, world_up, looking_at, view_size=(width, height), fx=fx)
    hit_pos = jax.vmap(partial(raycast, sdf, pos0, step_n))(ray_dir)
    mask = (jax.vmap(sdf)(hit_pos) < 1.0).astype(jnp.float32)
    return jnp.concatenate((hit_pos.reshape(height, width, 3) % 1.0, mask.reshape(height, width, 1)), axis=-1)
    # vis_data = jax.vmap(jax.grad(sdf))(hit_pos)
    # # vis_data = norm(jax.vmap(jax.grad(sdf))(hit_pos))[:,jnp.newaxis] * jnp.array([[1., 1., 1.]])
    # return vis_data.reshape(height, width, 3)

