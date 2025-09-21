import threading
import queue
import argparse
from pathlib import Path
import importlib.util

import numpy as np
from numpy.typing import ArrayLike
import OpenGL.GL as gl
import glfw
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from IPython import embed

from topax._utils import compile_shader, rotation_matrix_about_vector, normalize
from topax._shaders import ShaderGLSL
from topax.sdfs import SDF, empty

_SDF_REGISTRY = []
def show_part(sdf: SDF, color: ArrayLike):
    global _SDF_REGISTRY
    _SDF_REGISTRY.append(dict(sdf=sdf, color=color))
    return sdf

class SceneHandler:
    def __init__(self, window):
        self._window = window
        self._fb_width, self._fb_height = glfw.get_framebuffer_size(window)
        self._camera_position = np.array([0.0, 0.0, 1.0])
        self._camera_up = np.array([0.0, 1.0, 0.0])
        self._looking_at = np.array([0.0, 0.0, 0.0])
        self._fx = 1.0
        self.shader = ShaderGLSL()

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
        

    def draw_scene(self):
        """
        This function is responsible for drawing all parts of the scene. It will take in the 
        """
        gl.glViewport(0, 0, self._fb_width, self._fb_height)
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.shader.draw(self._fb_width, self._fb_height, self._camera_position, self._looking_at, self._camera_up, self._fx)

        glfw.swap_buffers(self._window)

class CLI:
    class FileEventHandler(FileSystemEventHandler):
        def __init__(self, callback):
            self._on_modified = callback

        def on_modified(self, event: FileSystemEvent) -> None:
            self._on_modified(event)

    def __init__(self, root_path, sdf_queue: queue.Queue):
        self._root_path = root_path
        self._target = None
        self._sdf_queue = sdf_queue
        self._thread = threading.Thread(target=self.repl_worker, daemon=True)
        self._event_handler = CLI.FileEventHandler(self._file_change_event)
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
        glfw.post_empty_event()

    def _file_change_event(self, event):
        if self._target is None: return
        try:
            if Path(event.src_path).samefile(self._target):
                self._sdf_queue.put(self._target)
                glfw.post_empty_event()
        except FileNotFoundError as e:
            pass

def main():
    global _SDF_REGISTRY
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
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "TOPAX", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

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
                scene.draw_scene()

    def scroll_callback(win, xoffset, yoffset):
        scene.zoom(yoffset)
        scene.draw_scene()

    def framebuffer_size_callback(win, width, height):
        scene._fb_width, scene._fb_height = glfw.get_framebuffer_size(window)
        scene.draw_scene()

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # Setup command message queue and project interface
    sdf_queue = queue.Queue()
    cli_interface = CLI(project_dir, sdf_queue)

    scene.draw_scene()

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
                scene.draw_scene()

        while not sdf_queue.empty():
            print("sdf queue item")
            new_sdf = sdf_queue.get()
            # sys.modules['cax.sdfs'] = cax.sdfs
            spec = importlib.util.spec_from_file_location("_external_script", new_sdf)
            module = importlib.util.module_from_spec(spec)
            _SDF_REGISTRY = []
            spec.loader.exec_module(module)
            scene.shader.update_sdfs([p["sdf"] for p in _SDF_REGISTRY], [p["color"] for p in _SDF_REGISTRY])
            scene.draw_scene()
            
        glfw.wait_events()

    # Clean up after app closes
    glfw.terminate()
    del scene

if __name__ == "__main__":
    main()
