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
import runpy

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from cax._utils import redraw_ortho, rotation_matrix, rotation_matrix_about_vector, _dummy_sdf, normalize

class SceneHandler:
    def __init__(self, window):
        self._window = window
        self._fb_width, self._fb_height = glfw.get_framebuffer_size(window)
        self._frame_buf = None
        self._texture_id = None
        self._sdf = _dummy_sdf()
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
    
    def _draw_view_cube(self, size=1.0):
        vp_size = 240
        gl.glViewport(0, self._fb_height-vp_size, vp_size, vp_size)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # glu.gluPerspective(35, 1.0, 0.1, 10)
        gl.glOrtho(-1, 1, -1, 1, -1, 10)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(*(self._camera_position / jnp.linalg.norm(self._camera_position)), 0,0,0, *self._camera_up)

        # apply same rotation as scene
        hs = size / 2
        # 6 colored faces
        faces = [
            ((1,0,0), [(hs,-hs,-hs),(hs,hs,-hs),(hs,hs,hs),(hs,-hs,hs)]),   # +X red
            ((0,1,0), [(-hs,-hs,hs),(-hs,hs,hs),(-hs,hs,-hs),(-hs,-hs,-hs)]), # -X green
            ((0,0,1), [(-hs,hs,hs),(hs,hs,hs),(hs,hs,-hs),(-hs,hs,-hs)]),   # +Y blue
            ((1,1,0), [(-hs,-hs,-hs),(hs,-hs,-hs),(hs,-hs,hs),(-hs,-hs,hs)]), # -Y yellow
            ((1,0,1), [(-hs,-hs,hs),(hs,-hs,hs),(hs,hs,hs),(-hs,hs,hs)]),   # +Z magenta
            ((0,1,1), [(-hs,hs,-hs),(hs,hs,-hs),(hs,-hs,-hs),(-hs,-hs,-hs)]) # -Z cyan
        ]
        gl.glEnable(gl.GL_DEPTH_TEST)
        for color, verts in faces:
            gl.glColor3f(*color)
            gl.glBegin(gl.GL_QUADS)
            for v in verts:
                gl.glVertex3f(*v)
            gl.glEnd()
    
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
        factor = (1 + delta * 0.008)
        self._fx *= factor
        self._camera_position *= factor
        

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
        
        gl.glViewport(0, 0, self._fb_width, self._fb_height)
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
        gl.glDisable(gl.GL_DEPTH_TEST)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1, 1, -1, 1, -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

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

        self._draw_view_cube()

        glfw.swap_buffers(self._window)

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self._on_modified = callback

    def on_modified(self, event: FileSystemEvent) -> None:
        self._on_modified(event)

class ProjectInterface:
    def __init__(self, root_path, sdf_queue: queue.Queue):
        self._root_path = root_path
        self._target = None
        self._sdf_queue = sdf_queue
        self._thread = threading.Thread(target=self.repl_worker, daemon=True)
        self._event_handler = FileEventHandler(self._file_change_event)
        self._observer = Observer()
        self._observer.schedule(self._event_handler, self._root_path, recursive=True)
        self._observer.start()
        self._thread.start()

    def repl_worker(self):
        banner = "CAX REPL started. Use shared_state/command_queue to talk to renderer."
        embed(header=banner, banner1="", colors="neutral", user_ns=dict(target=self.set_target_file))

    def set_target_file(self, path):
        path = Path(self._root_path, path)
        print("watching ", path)
        if not path.exists():
            print("targeted file doesn't exist!")
            return
        self._target = path
        self._sdf_queue.put(self._target)

    def _file_change_event(self, event):
        if self._target is None: return
        if Path(event.src_path).samefile(self._target):
            self._sdf_queue.put(self._target)
            glfw.post_empty_event()

def main():
    # Parse argments
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="top level directory of CAD project")
    args = parser.parse_args()
    project_dir = Path(args.dir)

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Can't find project dir {project_dir}")

    # Initialize glfw window
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    window = glfw.create_window(800, 600, "CAX", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Initialize scene handler
    scene = SceneHandler(window)

    # Initialize callbacks
    dragging = False
    last_pos_x, last_pos_y = 0, 0
    def mouse_button_callback(win, button, action, mods):
        nonlocal dragging, last_pos_x, last_pos_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            dragging = (action == glfw.PRESS)
            last_pos_x, last_pos_y = glfw.get_cursor_pos(window)
            if not dragging:
                scene.draw_scene(fast=False)

    def window_resize_callback(win, width, height):
        scene.update_fb_size()

    def scroll_callback(win, xoffset, yoffset):
        scene.zoom(yoffset)
        scene.draw_scene(fast=True)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_window_size_callback(window, window_resize_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Setup command message queue and project interface
    sdf_queue = queue.Queue()
    project_interface = ProjectInterface(project_dir, sdf_queue)

    scene.draw_scene(fast=False)

    # Main application loop
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

        while not sdf_queue.empty():
            new_sdf = sdf_queue.get()
            scene.update_sdf(runpy.run_path(str(new_sdf))["part1"])
            scene.draw_scene(fast=False)
            
        glfw.wait_events()

    # Clean up after app closes
    glfw.terminate()
    del scene

if __name__ == "__main__":
    main()
