import numpy as np
import OpenGL.GL as gl
from PIL import Image

def norm(v, axis=-1, keepdims=False, eps=0.0):
    return np.sqrt((v*v).sum(axis, keepdims=keepdims).clip(eps))

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

def compile_shader(src, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, src)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(shader).decode())
    return shader

def create_shader_program(vs, fs):
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(program).decode())
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return program

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
