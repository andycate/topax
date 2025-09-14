import argparse

import numpy as np
import OpenGL.GL as gl
import glfw

from topax._utils import (
    compile_shader, 
    normalize, 
    rotation_matrix_about_vector
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shader", help="shader source file")
    shader_file = parser.parse_args().shader
    with open(shader_file, "r") as f:
        shader_code = f.read()


    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Shader Viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    fb_width, fb_height = glfw.get_framebuffer_size(window)

    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    camera_position = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])
    looking_at = np.array([0.0, 0.0, 0.0])
    fx = 1.0

    shader_uniforms = dict(
        i_resolution=None,
        max_steps=None,
        cam_pose=None,
        looking_at=None,
        cam_up=None,
        fx=None,
        stop_epsilon=None,
        tmax=None
    )

    def rotate_2d(dx, dy):
        nonlocal camera_position, camera_up
        cam_right = normalize(np.linalg.cross(-camera_position, camera_up))
        x_rot = rotation_matrix_about_vector(-dx / 300., camera_up)
        y_rot = rotation_matrix_about_vector(-dy / 300., cam_right)
        camera_position = x_rot @ camera_position
        camera_position = y_rot @ camera_position
        camera_up = y_rot @ camera_up

    def zoom(delta):
        nonlocal fx, camera_position
        factor = (1 + delta * 0.008)
        fx *= factor
        camera_position *= factor

    vs = compile_shader(VERTEX_SHADER_SOURCE, gl.GL_VERTEX_SHADER)
    fs = compile_shader(shader_code, gl.GL_FRAGMENT_SHADER)
    program_id = gl.glCreateProgram()
    gl.glAttachShader(program_id, vs)
    gl.glAttachShader(program_id, fs)
    gl.glLinkProgram(program_id)
    if not gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program_id).decode())
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    gl.glUseProgram(program_id)

    shader_uniforms["i_resolution"] = gl.glGetUniformLocation(program_id, "_iResolution")
    shader_uniforms["max_steps"] = gl.glGetUniformLocation(program_id, "_maxSteps")
    shader_uniforms["cam_pose"] = gl.glGetUniformLocation(program_id, "_camPose")
    shader_uniforms["looking_at"] = gl.glGetUniformLocation(program_id, "_lookingAt")
    shader_uniforms["cam_up"] = gl.glGetUniformLocation(program_id, "_camUp")
    shader_uniforms["fx"] = gl.glGetUniformLocation(program_id, "_fx")
    shader_uniforms["stop_epsilon"] = gl.glGetUniformLocation(program_id, "_stopEpsilon")
    shader_uniforms["tmax"] = gl.glGetUniformLocation(program_id, "_tmax")

    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, QUAD.nbytes, QUAD, gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

    def draw_scene():
        """
        This function is responsible for drawing all parts of the scene. It will take in the 
        """
        nonlocal fb_width, fb_height, shader_uniforms, window, camera_position, camera_up, looking_at, fx
        gl.glViewport(0, 0, fb_width, fb_height)
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glUniform2f(shader_uniforms["i_resolution"], fb_width, fb_height)
        gl.glUniform1ui(shader_uniforms["max_steps"], 1024)
        gl.glUniform3f(shader_uniforms["cam_pose"], * camera_position)
        gl.glUniform3f(shader_uniforms["looking_at"], * looking_at)
        gl.glUniform3f(shader_uniforms["cam_up"], * camera_up)
        gl.glUniform1f(shader_uniforms["fx"], fx)
        gl.glUniform1f(shader_uniforms["stop_epsilon"], 0.00001)
        gl.glUniform1f(shader_uniforms["tmax"], 1000.0)

        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        glfw.swap_buffers(window)

    dragging = False
    last_pos_x, last_pos_y = 0, 0
    def mouse_button_callback(win, button, action, mods):
        nonlocal dragging, last_pos_x, last_pos_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            dragging = (action == glfw.PRESS)
            last_pos_x, last_pos_y = glfw.get_cursor_pos(window)
            if not dragging:
                draw_scene()

    def scroll_callback(win, xoffset, yoffset):
        zoom(yoffset)
        draw_scene()

    def window_resize_callback(win, width, height):
        global fb_width, fb_height
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        draw_scene()

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    draw_scene()

    while not glfw.window_should_close(window):
        if dragging:
            x, y = glfw.get_cursor_pos(window)
            dx = x - last_pos_x
            dy = y - last_pos_y
            last_pos_x = x
            last_pos_y = y

            if dx != 0 or dy != 0:
                rotate_2d(dx, dy)
                draw_scene()
            
        glfw.wait_events()

    # Clean up after app closes
    glfw.terminate()

if __name__ == "__main__":
    main()