import argparse
from dataclasses import dataclass
from typing import Any
from importlib import resources
from enum import IntEnum

from PIL import Image
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
import glm
import glfw

from topax._utils import (
    compile_shader, 
    create_shader_program,
    normalize, 
    rotation_matrix_about_vector
)

QUAD = np.array([
        -1.0, -1.0,
        1.0, -1.0,
        -1.0,  1.0,
        1.0,  1.0
    ], dtype=np.float32)

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;  // map [-1,1] -> [0,1]
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

VIEW_CUBE_VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 MVP;

out vec2 vTexCoord;

void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
"""

VIEW_CUBE_FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D cubeTexture;

void main() {
    FragColor = texture(cubeTexture, vTexCoord);
}
"""

class ShaderMode(IntEnum):
    AMBIENT=0
    MOD_ONE=1
    MARCH_STEPS=2

@dataclass
class AppState:
    fb_width: int = 0
    fb_height: int = 0
    camera_position: np.ndarray = None
    camera_up: np.ndarray = None
    looking_at: np.ndarray = None
    fx: float = 1.0
    program_id: int = 0
    vao: Any = None
    view_cube_vao: Any = None
    view_cube_program_id: int = 0
    view_cube_indices_len: int = 0
    view_cube_texture_id: int = 0
    last_mouse_pos_x = 0.0
    last_mouse_pos_y = 0.0
    last_mouse_button = 0
    mouse_dragging = False
    shader_mode = ShaderMode.AMBIENT

@dataclass
class ShaderUniforms:
    i_resolution = None
    max_steps = None
    cam_pose = None
    looking_at = None
    cam_up = None
    fx = None
    stop_epsilon = None
    tmax = None
    mode = None

state = AppState()
uniforms = ShaderUniforms()

def load_texture(file_name):
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
    
    # Set texture wrapping and filtering options
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    
    # Load image using Pillow
    try:
        image = Image.open(file_name)
        # OpenGL expects texture coordinates from bottom-left (0,0), but most images have (0,0) at top-left.
        # So we flip the image vertically.
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.convert("RGBA").tobytes()
        
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, image.width, image.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    except FileNotFoundError:
        print(f"Error: Texture file not found at '{path}'")
        return -1
        
    return texture_id

def create_cube_vao(size=1.0):
    hs = size / 2.0
    # vertices with colors per face

    vertices = [
        # pos             color
        ( hs,-hs,-hs,  0,   0), ( hs, hs,-hs,  1/6, 0), ( hs, hs, hs,  1/6, 1), ( hs,-hs, hs,  0,   1), # +X red (right)
        (-hs,-hs, hs,  2/6, 1), (-hs, hs, hs,  1/6, 1), (-hs, hs,-hs,  1/6, 0), (-hs,-hs,-hs,  2/6, 0), # -X green (left)
        (-hs, hs, hs,  3/6, 1), ( hs, hs, hs,  2/6, 1), ( hs, hs,-hs,  2/6, 0), (-hs, hs,-hs,  3/6, 0), # +Y blue (back)
        (-hs,-hs,-hs,  3/6, 0), ( hs,-hs,-hs,  4/6, 0), ( hs,-hs, hs,  4/6, 1), (-hs,-hs, hs,  3/6, 1), # -Y yellow (front)
        (-hs,-hs, hs,  4/6, 0), ( hs,-hs, hs,  5/6, 0), ( hs, hs, hs,  5/6, 1), (-hs, hs, hs,  4/6, 1), # +Z magenta (top)
        (-hs, hs,-hs,  5/6, 0), ( hs, hs,-hs,  6/6, 0), ( hs,-hs,-hs,  6/6, 1), (-hs,-hs,-hs,  5/6, 1)  # -Z cyan (bottom)
    ]
    vertices = np.array(vertices, dtype=np.float32)

    # indices (6 faces × 2 triangles each)
    indices = np.array([
        0,1,2,  2,3,0,
        4,5,6,  6,7,4,
        8,9,10, 10,11,8,
        12,13,14, 14,15,12,
        16,17,18, 18,19,16,
        20,21,22, 22,23,20
    ], dtype=np.uint32)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    stride = 5 * vertices.itemsize
    # position
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
    # color
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * vertices.itemsize))

    gl.glBindVertexArray(0)
    return vao, len(indices)

def draw_view_cube():
    global state
    vp_size = 240
    gl.glViewport(0, state.fb_height-vp_size, vp_size, vp_size)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    # Build projection + view
    projection = glm.ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 10.0)
    view = glm.lookAt(
        glm.normalize(glm.vec3(*state.camera_position)),
        glm.vec3(0,0,0),
        glm.vec3(*state.camera_up)
    )
    MVP = projection * view

    gl.glUseProgram(state.view_cube_program_id)
    loc = gl.glGetUniformLocation(state.view_cube_program_id, "MVP")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, np.array(MVP, dtype=np.float32).T)

    tex_loc = gl.glGetUniformLocation(state.view_cube_program_id, "cubeTexture")
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, state.view_cube_texture_id)
    gl.glUniform1i(tex_loc, 0)

    gl.glBindVertexArray(state.view_cube_vao)
    gl.glDrawElements(gl.GL_TRIANGLES, state.view_cube_indices_len, gl.GL_UNSIGNED_INT, None)
    gl.glBindVertexArray(0)

def draw_scene(fast=False):
    """
    This function is responsible for drawing all parts of the scene
    """
    global window, state, uniforms
    gl.glViewport(0, 0, state.fb_width, state.fb_height)
    gl.glClearColor(0.2, 0.2, 0.2, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    gl.glUseProgram(state.program_id)

    gl.glUniform2f(uniforms.i_resolution, state.fb_width, state.fb_height)
    gl.glUniform1ui(uniforms.max_steps, 128 if fast else 4096)
    gl.glUniform3f(uniforms.cam_pose, * state.camera_position)
    gl.glUniform3f(uniforms.looking_at, * state.looking_at)
    gl.glUniform3f(uniforms.cam_up, * state.camera_up)
    gl.glUniform1f(uniforms.fx, state.fx)
    gl.glUniform1f(uniforms.stop_epsilon, 0.01)
    gl.glUniform1f(uniforms.tmax, 30.0)
    gl.glUniform1ui(uniforms.mode, state.shader_mode)

    gl.glBindVertexArray(state.vao)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    draw_view_cube()

    glfw.swap_buffers(window)


def mouse_button_callback(win, button, action, mods):
    global window, state
    state.last_mouse_pos_x, state.last_mouse_pos_y = glfw.get_cursor_pos(window)
    state.mouse_dragging = (action == glfw.PRESS)
    state.last_mouse_button = button
    if not state.mouse_dragging:
        draw_scene()

def cursor_pos_callback(win, xpos, ypos):
    global window, state
    if state.mouse_dragging:
        if state.last_mouse_button == glfw.MOUSE_BUTTON_LEFT:
            dx = xpos - state.last_mouse_pos_x
            dy = ypos - state.last_mouse_pos_y
            state.last_mouse_pos_x = xpos
            state.last_mouse_pos_y = ypos
            cam_right = normalize(np.linalg.cross(-state.camera_position, state.camera_up))
            x_rot = rotation_matrix_about_vector(-dx / 300.0, state.camera_up)
            y_rot = rotation_matrix_about_vector(-dy / 300.0, cam_right)
            state.camera_position = x_rot @ state.camera_position
            state.camera_position = y_rot @ state.camera_position
            state.camera_up = y_rot @ state.camera_up
            draw_scene(fast=True)
        elif state.last_mouse_button == glfw.MOUSE_BUTTON_RIGHT:
            dx = xpos - state.last_mouse_pos_x
            dy = ypos - state.last_mouse_pos_y
            state.last_mouse_pos_x = xpos
            state.last_mouse_pos_y = ypos
            cam_right = normalize(np.linalg.cross(-state.camera_position, state.camera_up))
            state.looking_at += state.camera_up * dy * 1e-3
            state.looking_at -= cam_right * dx * 1e-3
            draw_scene(fast=True)

def scroll_callback(win, xoffset, yoffset):
    global state
    factor = (1 + yoffset * 0.008)
    state.fx *= factor
    state.camera_position *= factor
    draw_scene(fast=True)

def window_resize_callback(win, width, height):
    global window, state
    state.fb_width, state.fb_height = glfw.get_framebuffer_size(window)
    draw_scene(fast=True)

def key_callback(_window, key, _scan, action, _mods):
    global state
    if action == glfw.PRESS and key == glfw.KEY_M:
        state.shader_mode = (state.shader_mode + 1) % len(ShaderMode)
        draw_scene()

def main():
    global window, state, uniforms

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("shader", help="shader source file")
    shader_file = parser.parse_args().shader
    with open(shader_file, "r") as f:
        shader_code = f.read()


    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    # Required for MacOS OpenGL to work
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    state.fb_width, state.fb_height = glfw.get_framebuffer_size(window)

    state.vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    

    state.camera_position = np.array([-1.0, -2.0, 1.0])
    state.camera_up = np.cross(np.cross(state.camera_position, np.array([0.0, 0.0, 1.0])), state.camera_position)
    state.camera_up /= np.linalg.norm(state.camera_up)
    state.looking_at = np.array([0.0, 0.0, 0.0])
    state.fx = 3.0

    # compile and load view_cube shader program
    vs = compile_shader(VIEW_CUBE_VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
    fs = compile_shader(VIEW_CUBE_FRAGMENT_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
    state.view_cube_program_id = create_shader_program(vs, fs)

    state.view_cube_texture_id = load_texture("topax/resources/viewcube.png")
    state.view_cube_vao, state.view_cube_indices_len = create_cube_vao()

    # compile and load shader program
    vs = compile_shader(VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
    fs = compile_shader(shader_code, gl.GL_FRAGMENT_SHADER)
    state.program_id = create_shader_program(vs, fs)
    gl.glUseProgram(state.program_id)

    # get shader uniform locations
    uniforms.i_resolution = gl.glGetUniformLocation(state.program_id, "_iResolution")
    uniforms.max_steps = gl.glGetUniformLocation(state.program_id, "_maxSteps")
    uniforms.cam_pose = gl.glGetUniformLocation(state.program_id, "_camPose")
    uniforms.looking_at = gl.glGetUniformLocation(state.program_id, "_lookingAt")
    uniforms.cam_up = gl.glGetUniformLocation(state.program_id, "_camUp")
    uniforms.fx = gl.glGetUniformLocation(state.program_id, "_fx")
    uniforms.stop_epsilon = gl.glGetUniformLocation(state.program_id, "_stopEpsilon")
    uniforms.tmax = gl.glGetUniformLocation(state.program_id, "_tmax")
    uniforms.mode = gl.glGetUniformLocation(state.program_id, "_mode")

    # draw the vertices that cover the whole screen
    gl.glBindVertexArray(state.vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, QUAD.nbytes, QUAD, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

    # set up event callback functions
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_size_callback(window, window_resize_callback)
    glfw.set_key_callback(window, key_callback)

    draw_scene() # initial draw

    # run the event loop
    while not glfw.window_should_close(window):
        glfw.wait_events()

    glfw.terminate()

if __name__ == "__main__":
    main()