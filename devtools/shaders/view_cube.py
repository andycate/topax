import numpy as np
from OpenGL import GL as gl
import glm  # pip install PyGLM
from topax._utils import compile_shader, create_shader_program

# ---- SHADERS ----
VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 MVP;

out vec3 vColor;

void main() {
    vColor = aColor;
    gl_Position = MVP * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
"""

vs = compile_shader(VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
fs = compile_shader(FRAGMENT_SHADER_SRC, gl.GL_FRAGMENT_SHADER)

# ---- GEOMETRY ----
def create_cube(size=1.0):
    hs = size / 2.0
    # vertices with colors per face
    vertices = [
        # pos             color
        ( hs,-hs,-hs,  1,0,0), ( hs, hs,-hs,  1,0,0), ( hs, hs, hs,  1,0,0), ( hs,-hs, hs,  1,0,0), # +X red
        (-hs,-hs, hs,  0,1,0), (-hs, hs, hs,  0,1,0), (-hs, hs,-hs,  0,1,0), (-hs,-hs,-hs,  0,1,0), # -X green
        (-hs, hs, hs,  0,0,1), ( hs, hs, hs,  0,0,1), ( hs, hs,-hs,  0,0,1), (-hs, hs,-hs,  0,0,1), # +Y blue
        (-hs,-hs,-hs,  1,1,0), ( hs,-hs,-hs,  1,1,0), ( hs,-hs, hs,  1,1,0), (-hs,-hs, hs,  1,1,0), # -Y yellow
        (-hs,-hs, hs,  1,0,1), ( hs,-hs, hs,  1,0,1), ( hs, hs, hs,  1,0,1), (-hs, hs, hs,  1,0,1), # +Z magenta
        (-hs, hs,-hs,  0,1,1), ( hs, hs,-hs,  0,1,1), ( hs,-hs,-hs,  0,1,1), (-hs,-hs,-hs, 0,1,1)  # -Z cyan
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

    return vertices, indices

def create_cube_vao(vertices, indices):
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    ebo = gl.glGenBuffers(1)

    gl.glBindVertexArray(vao)

    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

    stride = 6 * vertices.itemsize
    # position
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
    # color
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(3 * vertices.itemsize))

    gl.glBindVertexArray(0)
    return vao, vbo, ebo

# ---- DRAW ----
def draw_view_cube(program, vao, indices, state, size=1.0):
    vp_size = 240
    gl.glViewport(0, state.fb_height-vp_size, vp_size, vp_size)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    # Build projection + view
    projection = glm.ortho(-1, 1, -1, 1, -1, 10)
    view = glm.lookAt(
        glm.normalize(glm.vec3(*state.camera_position)),
        glm.vec3(0,0,0),
        glm.vec3(*state.camera_up)
    )
    MVP = projection * view

    gl.glUseProgram(program)
    loc = gl.glGetUniformLocation(program, "MVP")
    gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, np.array(MVP, dtype=np.float32))

    gl.glBindVertexArray(vao)
    gl.glDrawElements(gl.GL_TRIANGLES, len(indices), gl.GL_UNSIGNED_INT, None)
    gl.glBindVertexArray(0)
