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

def _dummy_sdf():
    def _df(p):
        return jnp.inf
    return _df

def raycast_ortho(sdf, dir, step_n, p0):
    def f(_, p):
        return p+sdf(p)*dir
    return jax.lax.fori_loop(0, step_n, f, p0)

def camera_rays_ortho(w, h, camera_position, camera_up, looking_at, fx):
    cam_norm = looking_at - camera_position
    right = jnp.cross(cam_norm, camera_up)
    down = jnp.cross(right, cam_norm)
    R = normalize(jnp.vstack([right, down]))
    fy = fx/w*h
    y, x = jnp.mgrid[-1.:1.:h*1j, -1.:1.:w*1j].reshape(2, -1)
    y *= fy
    x *= fx
    return jnp.c_[x, y] @ R + camera_position.reshape(1, 3)

@partial(jax.jit, static_argnames=("sdf", "w", "h", "step_n"))
def redraw_ortho(sdf, w, h, step_n, camera_position, camera_up, looking_at, fx):
    ray_pos = camera_rays_ortho(w, h, camera_position, camera_up, looking_at, fx)
    hit_pos = jax.vmap(partial(raycast_ortho, sdf, normalize(looking_at - camera_position), step_n))(ray_pos)
    mask = (jax.vmap(sdf)(hit_pos) < 1.0).astype(jnp.float32)
    # mask = jnp.ones(hit_pos.shape[0])
    # shading = jax.vmap(jax.grad(sdf))(hit_pos)
    # shading = jnp.abs(shading)
    return jnp.concatenate((hit_pos.reshape(h, w, 3) % 1.0, mask.reshape(h, w, 1)), axis=-1)
    # return jnp.concatenate((shading, mask[:, jnp.newaxis]), axis=-1).reshape(h, w, 4)
