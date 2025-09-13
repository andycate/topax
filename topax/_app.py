import threading
import queue
import argparse
from pathlib import Path
import importlib.util

import numpy as np
import OpenGL.GL as gl
import glfw
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from IPython import embed

from topax._utils import compile_shader, rotation_matrix_about_vector, normalize
from topax._builders import Builder
from topax.sdfs import empty
from topax.ops import Const

class SceneHandler:
    QUAD = np.array([
        -1.0, -1.0,
        1.0, -1.0,
        -1.0,  1.0,
        1.0,  1.0
    ], dtype=np.float32)
    VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;  // map [-1,1] -> [0,1]
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""
    def __init__(self, window):
        self._window = window
        self._fb_width, self._fb_height = glfw.get_framebuffer_size(window)
        self._program_id = None
        self._shader_source = ""
        self._sdf_repr = None
        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        self._camera_position = np.array([0.0, 0.0, 1.0])
        self._camera_up = np.array([0.0, 1.0, 0.0])
        self._looking_at = np.array([0.0, 0.0, 0.0])
        self._fx = 1.0

        self._shader_uniforms = dict(
            i_resolution=None,
            max_steps=None,
            cam_pose=None,
            looking_at=None,
            cam_up=None,
            fx=None,
            stop_epsilon=None,
            tmax=None
        )

        self.update_sdf(empty())

    def __del__(self):
        if self._program_id is not None:
            gl.glDeleteProgram(self._program_id)
        # gl.glDeleteBuffers(self._vbo)
        # gl.glDeleteVertexArrays(self._vao)
    
    def update_sdf(self, sdf):
        sdf_repr = repr(sdf(Const(None, 'p', 'vec3')))
        builder = Builder(sdf)
        if sdf_repr != self._sdf_repr:
            print("recompiling shader!")
            self._sdf_repr = sdf_repr
            shader_code = builder.build()
            input_vars = builder.get_input_vars()

            if self._program_id is not None:
                gl.glDeleteProgram(self._program_id)
            
            vs = compile_shader(SceneHandler.VERTEX_SHADER_SOURCE, gl.GL_VERTEX_SHADER)
            fs = compile_shader(shader_code, gl.GL_FRAGMENT_SHADER)
            self._program_id = gl.glCreateProgram()
            gl.glAttachShader(self._program_id, vs)
            gl.glAttachShader(self._program_id, fs)
            gl.glLinkProgram(self._program_id)
            if not gl.glGetProgramiv(self._program_id, gl.GL_LINK_STATUS):
                raise RuntimeError(gl.glGetProgramInfoLog(self._program_id).decode())
            gl.glDeleteShader(vs)
            gl.glDeleteShader(fs)
            gl.glUseProgram(self._program_id)

            self._shader_uniforms["i_resolution"] = gl.glGetUniformLocation(self._program_id, "_iResolution")
            self._shader_uniforms["max_steps"] = gl.glGetUniformLocation(self._program_id, "_maxSteps")
            self._shader_uniforms["cam_pose"] = gl.glGetUniformLocation(self._program_id, "_camPose")
            self._shader_uniforms["looking_at"] = gl.glGetUniformLocation(self._program_id, "_lookingAt")
            self._shader_uniforms["cam_up"] = gl.glGetUniformLocation(self._program_id, "_camUp")
            self._shader_uniforms["fx"] = gl.glGetUniformLocation(self._program_id, "_fx")
            self._shader_uniforms["stop_epsilon"] = gl.glGetUniformLocation(self._program_id, "_stopEpsilon")
            self._shader_uniforms["tmax"] = gl.glGetUniformLocation(self._program_id, "_tmax")
        else:
            builder.parse_input_vars()
            input_vars = builder.get_input_vars()

        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, SceneHandler.QUAD.nbytes, SceneHandler.QUAD, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        for input_var, param in input_vars.items():
            loc = gl.glGetUniformLocation(self._program_id, input_var)
            value = param.resolve_value()
            match param.rettype:
                case "vec3":
                    gl.glUniform3f(loc, *value)
                case "vec2":
                    gl.glUniform2f(loc, *value)
                case "float":
                    if hasattr(value, "__iter__"):
                        value = float(value[0])
                    else:
                        value = float(value)
                    gl.glUniform1f(loc, value)
                case _:
                    raise NotImplementedError()

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

        gl.glUniform2f(self._shader_uniforms["i_resolution"], self._fb_width, self._fb_height)
        gl.glUniform1ui(self._shader_uniforms["max_steps"], 256)
        gl.glUniform3f(self._shader_uniforms["cam_pose"], *self._camera_position)
        gl.glUniform3f(self._shader_uniforms["looking_at"], *self._looking_at)
        gl.glUniform3f(self._shader_uniforms["cam_up"], *self._camera_up)
        gl.glUniform1f(self._shader_uniforms["fx"], self._fx)
        gl.glUniform1f(self._shader_uniforms["stop_epsilon"], 0.0001)
        gl.glUniform1f(self._shader_uniforms["tmax"], 1000.0)

        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

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

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Setup command message queue and project interface
    sdf_queue = queue.Queue()
    project_interface = ProjectInterface(project_dir, sdf_queue)

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
            spec.loader.exec_module(module)
            scene.update_sdf(module.make_part())
            scene.draw_scene()
            
        glfw.wait_events()

    # Clean up after app closes
    glfw.terminate()
    del scene

if __name__ == "__main__":
    main()
