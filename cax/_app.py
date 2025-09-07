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
import argparse
from pathlib import Path

from cax._utils import redraw_ortho, rotation_matrix, rotation_matrix_about_vector, _dummy_sdf, normalize
from cax.sdfs import test_sdf

class SceneHandler:
    def __init__(self, window):
        self._window = window
        self._fb_width, self._fb_height = glfw.get_framebuffer_size(window)
        self._frame_buf = None
        self._texture_id = None
        self._sdf = test_sdf()
        self._hifi_render = None
        self._lofi_render = None
        self._camera_position = np.array([0.0, 0.0, 1.0])
        self._camera_up = np.array([0.0, 1.0, 0.0])
        self._looking_at = np.array([0.0, 0.0, 0.0])
        self._fx = 1.0

        self._make_texture()
        self._gen_render_funcs()

    def __del__(self):
        gl.glDeleteTextures(1, [self._texture_id])

    def _gen_render_funcs(self):
        self._hifi_render = redraw_ortho.trace(self._sdf, self._fb_width, self._fb_height, 100, self._camera_position, self._camera_up, self._looking_at, self._fx).lower().compile()
        self._lofi_render = redraw_ortho.trace(self._sdf, self._fb_width, self._fb_height, 30, self._camera_position, self._camera_up, self._looking_at, self._fx).lower().compile()

    def _make_texture(self):
        self._frame_buf = np.zeros((self._fb_height, self._fb_width, 4), dtype=np.float32)
        gl.glViewport(0, 0, self._fb_width, self._fb_height)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        # Generate a new texture
        if self._texture_id is None:
            self._texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)

        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            self._fb_width,
            self._fb_height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            self._frame_buf,
        )
        return self._texture_id
    
    def update_sdf(self, sdf):
        self._sdf = sdf
        self._gen_render_funcs()

    def update_fb_size(self):
        self._fb_width, self._fb_height = glfw.get_framebuffer_size(self._window)
        self._make_texture()
        self._gen_render_funcs()

    def rotate_2d(self, dx, dy):
        cam_right = normalize(np.linalg.cross(-self._camera_position, self._camera_up))
        x_rot = rotation_matrix_about_vector(-dx / 300., self._camera_up)
        y_rot = rotation_matrix_about_vector(-dy / 300., cam_right)
        self._camera_position = x_rot @ self._camera_position
        self._camera_position = y_rot @ self._camera_position
        self._camera_up = y_rot @ self._camera_up

    def zoom(self, delta):
        self._fx *= (1+delta * 0.008)
        

    def draw_scene(self, fast=False):
        """
        This function is responsible for drawing all parts of the scene. It will take in the 
        """
        if fast:
            np.copyto(
                self._frame_buf, 
                np.array(
                    self._lofi_render(self._camera_position, self._camera_up, self._looking_at, self._fx), 
                    copy=False, 
                    dtype=np.float32
                )
            )
        else:
            np.copyto(
                self._frame_buf, 
                np.array(
                    self._hifi_render(self._camera_position, self._camera_up, self._looking_at, self._fx), 
                    copy=False, 
                    dtype=np.float32
                )
            )
        
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0, 0,
            self._fb_width, self._fb_height,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            self._frame_buf,
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

        glfw.swap_buffers(self._window)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="top level directory of CAD project")
    args = parser.parse_args()
    project_dir = Path(args.dir)

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Can't find project dir {project_dir}")

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(800, 600, "CAX", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    scene = SceneHandler(window)

    dragging = False
    last_pos_x, last_pos_y = 0, 0
    def mouse_button_callback(win, button, action, mods):
        nonlocal dragging, last_pos_x, last_pos_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            dragging = (action == glfw.PRESS)
            last_pos_x, last_pos_y = glfw.get_cursor_pos(window)
            if not dragging:
                scene.draw_scene(fast=False)

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    def window_resize_callback(win, width, height):
        scene.update_fb_size()

    glfw.set_window_size_callback(window, window_resize_callback)

    def scroll_callback(win, xoffset, yoffset):
        scene.zoom(yoffset)
        scene.draw_scene(fast=True)

    glfw.set_scroll_callback(window, scroll_callback)

    while not glfw.window_should_close(window):
        if dragging:
            x, y = glfw.get_cursor_pos(window)
            dx = x - last_pos_x
            dy = y - last_pos_y
            last_pos_x = x
            last_pos_y = y

            if dx != 0 or dy != 0:
                scene.rotate_2d(dx, dy)
                scene.draw_scene(fast=True)
            
        glfw.wait_events()

    glfw.terminate()
    del scene

if __name__ == "__main__":
    main()
