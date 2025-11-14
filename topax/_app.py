import threading
import queue
import argparse
from pathlib import Path
import importlib.util
from dataclasses import dataclass
import tkinter as tk
import multiprocessing as mp

import numpy as np
from numpy.typing import ArrayLike
import OpenGL.GL as gl
import glfw
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
import pyspacemouse

from topax._utils import rotation_matrix_about_vector, normalize
from topax._shaders import ShaderGLSL, ShaderMode
from topax.sdfs import SDF

@dataclass
class SDFRegistryEntry:
    sdf: SDF
    color: ArrayLike

_SDF_REGISTRY = []
def show_part(sdf: SDF, color: ArrayLike):
    global _SDF_REGISTRY
    _SDF_REGISTRY.append(SDFRegistryEntry(sdf, color))
    return sdf

class SceneHandler:
    def __init__(self, window, print_code=False):
        self.window = window
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(window)
        self.camera_position = np.array([0.0, -1.0, 0.0])
        self.camera_up = np.array([0.0, 0.0, 1.0])
        self.looking_at = np.array([0.0, 0.0, 0.0])
        self.fx = 1.0
        self.mode = ShaderMode.AMBIENT
        self.shader = ShaderGLSL(print_code=print_code)

    def set_view_front(self):
        self.camera_position = np.array([0.0, -1.0, 0.0]) * np.linalg.norm(self.camera_position)
        self.camera_up = np.array([0.0, 0.0, 1.0])
        self.looking_at = np.array([0.0, 0.0, 0.0])

    def set_view_top(self):
        self.camera_position = np.array([0.0, 0.0, 1.0]) * np.linalg.norm(self.camera_position)
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.looking_at = np.array([0.0, 0.0, 0.0])

    def set_view_left(self):
        self.camera_position = np.array([-1.0, 0.0, 0.0]) * np.linalg.norm(self.camera_position)
        self.camera_up = np.array([0.0, 0.0, 1.0])
        self.looking_at = np.array([0.0, 0.0, 0.0])

    def set_view_right(self):
        self.camera_position = np.array([1.0, 0.0, 0.0]) * np.linalg.norm(self.camera_position)
        self.camera_up = np.array([0.0, 0.0, 1.0])
        self.looking_at = np.array([0.0, 0.0, 0.0])

    def rotate_2d(self, dx, dy):
        cam_right = normalize(np.linalg.cross(-self.camera_position, self.camera_up))
        x_rot = rotation_matrix_about_vector(-dx / 300., self.camera_up)
        y_rot = rotation_matrix_about_vector(-dy / 300., cam_right)
        self.camera_position = x_rot @ self.camera_position
        self.camera_position = y_rot @ self.camera_position
        self.camera_up = y_rot @ self.camera_up

    def rotate_rpy(self, roll, pitch, yaw):
        cam_right = normalize(np.linalg.cross(-self.camera_position, self.camera_up))
        x_rot = rotation_matrix_about_vector(yaw * 0.04, self.camera_up)
        y_rot = rotation_matrix_about_vector(pitch * 0.04, cam_right)
        z_rot = rotation_matrix_about_vector(roll * 0.04, -self.camera_position)
        self.camera_position = x_rot @ self.camera_position
        self.camera_position = y_rot @ self.camera_position
        self.camera_position = z_rot @ self.camera_position
        self.camera_up = y_rot @ self.camera_up
        self.camera_up = z_rot @ self.camera_up

    def zoom(self, delta):
        factor = (1 + delta * 0.008)
        self.fx *= factor
        self.camera_position *= factor
        

    def draw_scene(self, fast=False):
        """
        This function is responsible for drawing all parts of the scene. It will take in the 
        """
        gl.glViewport(0, 0, self.fb_width, self.fb_height)
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if fast:
            steps = 128
            stop_epsilon = 0.001
            tmax = 100.0
        else:
            steps = 1024
            stop_epsilon = 0.00001
            tmax = 500.0

        self.shader.draw(
            self.fb_width, 
            self.fb_height,
            self.camera_position, 
            self.looking_at, 
            self.camera_up, 
            self.fx, 
            stop_epsilon,
            tmax,
            self.mode,
            steps
        )

        glfw.swap_buffers(self.window)


class CLI:
    class FileEventHandler(FileSystemEventHandler):
        def __init__(self, callback):
            self._on_modified = callback

        def on_modified(self, event: FileSystemEvent) -> None:
            self._on_modified(event)

    def __init__(self, target_path, sdf_event: threading.Event):
        self.target_path = target_path
        self.sdf_event = sdf_event
        self.event_handler = CLI.FileEventHandler(self._file_change_event)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, Path(self.target_path).parent, recursive=True)
        self.observer.start()

    def _file_change_event(self, event):
        if not Path(event.src_path).exists(): return
        if Path(event.src_path).samefile(self.target_path):
            self.sdf_event.set()
            glfw.post_empty_event()


def param_process_func(def_q: mp.Queue, update_q: mp.Queue, param_change_event):
    app = tk.Tk()

    def get_on_change_func(label):
        def on_change(val):
            if update_q.empty():
                update_q.put((label, val))
                param_change_event.set()
        return on_change

    label_control_pairs = []

    def process_def_q():
        nonlocal label_control_pairs, app, def_q
        if not def_q.empty():
            param_stack = def_q.get()
            for lc in label_control_pairs:
                lc[0].grid_forget()
                lc[0].destroy()
                lc[1].grid_forget()
                lc[1].destroy()
            label_control_pairs.clear()
            for i, ps in enumerate(param_stack):
                label = tk.Label(app, text=ps['label'])
                label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
                slider = tk.Scale(app, orient="horizontal", resolution=ps['resolution'], length=200, from_=ps['from'], to=ps['to'], command=get_on_change_func(ps['label']))
                slider.set(ps['value'])
                slider.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
                label_control_pairs.append((label, slider))
            app.grid_columnconfigure(1, weight=1)
        app.after(20, process_def_q)

    process_def_q()
    app.mainloop()


def main():
    global _SDF_REGISTRY
    # Parse argments
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacemouse", action='store_true', help='enable space mouse interface')
    parser.add_argument("--auto_reload", action='store_true', help='enable auto file reloading')
    parser.add_argument("--print_code", action='store_true', help='enable printing shader code on recompile')
    parser.add_argument("file", help="python file to read from")
    args = parser.parse_args()
    project_file = Path(args.file)

    if not project_file.exists():
        raise FileNotFoundError(f"Can't find project file {project_file}")

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
    scene = SceneHandler(window, args.print_code)

    param_def_queue = mp.Queue()
    param_change_queue = mp.Queue()
    param_change_event = mp.Event()
    param_process = mp.Process(target=param_process_func, args=(param_def_queue, param_change_queue, param_change_event), daemon=True)

    # Initialize callbacks
    mouse_dragging = False
    last_mouse_button = None
    last_pos_x, last_pos_y = 0, 0
    def mouse_button_callback(win, button, action, mods):
        nonlocal mouse_dragging, last_pos_x, last_pos_y, last_mouse_button
        mouse_dragging = (action == glfw.PRESS)
        last_pos_x, last_pos_y = glfw.get_cursor_pos(window)
        last_mouse_button = button
        if not mouse_dragging:
            scene.draw_scene()

    def cursor_pos_callback(win, xpos, ypos):
        nonlocal window, mouse_dragging, last_pos_x, last_pos_y, last_mouse_button, scene
        if mouse_dragging:
            if last_mouse_button == glfw.MOUSE_BUTTON_LEFT:
                dx = xpos - last_pos_x
                dy = ypos - last_pos_y
                last_pos_x = xpos
                last_pos_y = ypos
                scene.rotate_2d(dx, dy)
                scene.draw_scene(fast=True)

    def scroll_callback(win, xoffset, yoffset):
        scene.zoom(yoffset)
        scene.draw_scene(fast=True)

    def framebuffer_size_callback(win, width, height):
        scene.fb_width, scene.fb_height = glfw.get_framebuffer_size(window)
        scene.draw_scene()

    def update_target_file(def_queue: mp.Queue):
        global _SDF_REGISTRY
        nonlocal scene, args
        spec = importlib.util.spec_from_file_location("_external_script", args.file)
        module = importlib.util.module_from_spec(spec)
        _SDF_REGISTRY = []
        spec.loader.exec_module(module)
        ext_params = scene.shader.update_sdfs([p.sdf for p in _SDF_REGISTRY], [p.color for p in _SDF_REGISTRY])
        def_queue.put([{
            'label': p.name,
            'resolution': p.resolution,
            'from': p.vmin,
            'to': p.vmax,
            'value': p.value
        } for p in ext_params])
        scene.draw_scene()

    def key_callback(_window, key, _scan, action, _mods):
        nonlocal scene
        if action == glfw.PRESS:
            match key:
                case glfw.KEY_M:
                    scene.mode = (scene.mode + 1) % len(ShaderMode)
                    scene.draw_scene()
                case glfw.KEY_U:
                    # reload target file
                    print("updating target file")
                    update_target_file(param_def_queue)
                case glfw.KEY_F:
                    # move view to front
                    print("setting view to front")
                    scene.set_view_front()
                    scene.draw_scene()
                case glfw.KEY_T:
                    # move view to front
                    print("setting view to top")
                    scene.set_view_top()
                    scene.draw_scene()
                case glfw.KEY_L:
                    # move view to front
                    print("setting view to left")
                    scene.set_view_left()
                    scene.draw_scene()
                case glfw.KEY_R:
                    # move view to front
                    print("setting view to right")
                    scene.set_view_right()
                    scene.draw_scene()


    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_key_callback(window, key_callback)



    spacemouse_queue = queue.Queue(1)
    def spacemouse_thread():
        while True: pyspacemouse.read()

    def spacemouse_update(data):
        spacemouse_queue.put(data)
        glfw.post_empty_event()

    # conditionally initialize space mouse
    if args.spacemouse:
        success = pyspacemouse.open(dof_callback=spacemouse_update)
        if not success: print("WARNING: can't open space mouse!")
        else:
            spacemouse_thread_id = threading.Thread(target=spacemouse_thread, daemon=True)
            spacemouse_thread_id.start()

    # set up auto file reloading
    sdf_file_change_event = threading.Event()
    if args.auto_reload:
        sdf_reloader_cli = CLI(args.file, sdf_file_change_event)

    param_process.start()


    def param_change_listener():
        while True:
            param_change_event.wait()
            param_change_event.clear()
            glfw.post_empty_event()

    param_change_listener_thread = threading.Thread(target=param_change_listener, daemon=True)
    param_change_listener_thread.start()

    update_target_file(param_def_queue)

    # Main application loop
    while not glfw.window_should_close(window):
        if not spacemouse_queue.empty():
            dof = spacemouse_queue.get()
            scene.rotate_rpy(dof.roll, dof.pitch, dof.yaw)
            scene.zoom(dof.y)
            scene.draw_scene(fast=True)

        if sdf_file_change_event.is_set():
            sdf_file_change_event.clear()
            update_target_file(param_def_queue)

        if not param_change_queue.empty():
            change = param_change_queue.get()
            scene.shader.update_explicit_param(change[0], change[1])
            scene.draw_scene(fast=False)

        glfw.wait_events()

    # Clean up after app closes
    glfw.terminate()
    del scene

if __name__ == "__main__":
    main()
