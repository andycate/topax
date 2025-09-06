import jax
import jax.numpy as jnp
import numpy as np

import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import glfw

from functools import partial
from typing import NamedTuple
import threading
import queue
import time
from IPython import embed

from cax._utils import redraw, rotation_matrix, rotation_matrix_about_vector
from cax.sdfs import test_sdf

# global defs
window_width, window_height = 800, 600
fb_width, fb_height = None, None
frame_buf = None # numpy array for texture
dragging = False
last_pos_x, last_pos_y = 0, 0
cam_pose = np.array([3.0, 4.0, 5.0]) * 0.5
looking_at = np.array([0., 0., 0.])
world_up = np.array([0., 0., 1.])
zoom = 0.8
sdf = test_sdf()

command_queue = queue.Queue()

def make_texture(texture_id=None):
    global frame_buf, fb_width, fb_height

    frame_buf = np.zeros((fb_height, fb_width, 4), dtype=np.float32)
    gl.glViewport(0, 0, fb_width, fb_height)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # Generate a new texture
    if texture_id is None:
        texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    # Upload the NumPy array data to the texture
    # The image is upside-down in OpenGL, so a flip is often needed.
    # PIL's `transpose` is one way, or you can handle it with the texture matrix.
    # Here we assume the array is already oriented correctly or will be corrected elsewhere.
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA32F,
        fb_width,
        fb_height,
        0,
        gl.GL_RGBA,
        gl.GL_FLOAT,
        frame_buf,
    )
    return texture_id

def draw_scene(texture_id, fast=False):
    global frame_buf, dragging, last_pos_x, last_pos_y, cam_pose, window_width, window_height, fb_width, fb_height, sdf, world_up, looking_at, zoom

    np.copyto(
        frame_buf, 
        np.array(
            redraw(sdf, fb_width, fb_height, 20 if fast else 100, cam_pose, world_up, looking_at, zoom, jnp.array([0.8, 0.2, 0.1])), 
            copy=False, 
            dtype=np.float32
        )
    )  # get new frame from JAX
    
    gl.glClearColor(0.2, 0.2, 0.2, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    gl.glTexSubImage2D(
        gl.GL_TEXTURE_2D,
        0,
        0, 0,
        fb_width, fb_height,
        gl.GL_RGBA,
        gl.GL_FLOAT,
        frame_buf,
    )

    # Bind the texture to display it
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glColor3f(1.0, 1.0, 1.0)

    # Draw a quad to display the texture
    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2f(0, 0)
    gl.glVertex2f(-1, -1)
    gl.glTexCoord2f(1, 0)
    gl.glVertex2f(1, -1)
    gl.glTexCoord2f(1, 1)
    gl.glVertex2f(1, 1)
    gl.glTexCoord2f(0, 1)
    gl.glVertex2f(-1, 1)
    gl.glEnd()

    gl.glDisable(gl.GL_TEXTURE_2D)

def main():
    global frame_buf, dragging, last_pos_x, last_pos_y, cam_pose, window_width, window_height, fb_width, fb_height, sdf, world_up, looking_at, zoom

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(window_width, window_height, "CAX", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    fb_width, fb_height = glfw.get_framebuffer_size(window)

    texture_id = make_texture()
    draw_scene(texture_id)
    glfw.swap_buffers(window)

    # Mouse state
    def mouse_button_callback(win, button, action, mods):
        global dragging, last_pos_x, last_pos_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            dragging = (action == glfw.PRESS)
            last_pos_x, last_pos_y = glfw.get_cursor_pos(window)
            if not dragging:
                draw_scene(texture_id, fast=False)
                glfw.swap_buffers(window)

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    def window_resize_callback(win, width, height):
        global fb_width, fb_height
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        gl.glViewport(0, 0, fb_width, fb_height)
        make_texture(texture_id)

    glfw.set_window_size_callback(window, window_resize_callback)

    while not glfw.window_should_close(window):
        if dragging:
            x, y = glfw.get_cursor_pos(window)
            dx = x - last_pos_x
            dy = y - last_pos_y
            last_pos_x = x
            last_pos_y = y

            if dx != 0 or dy != 0:
                world_side = np.linalg.cross(cam_pose, world_up)
                world_side_rot = rotation_matrix_about_vector(dy / 300., world_side)
                cam_pose = rotation_matrix_about_vector(-dx / 300., world_up) @ cam_pose
                cam_pose = world_side_rot @ cam_pose
                world_up = world_side_rot @ world_up

                draw_scene(texture_id, fast=True)
                glfw.swap_buffers(window)
            
        glfw.wait_events()

    glfw.terminate()
    gl.glDeleteTextures(1, [texture_id])

if __name__ == "__main__":
    main()
