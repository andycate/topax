from typing import Any
from collections import deque
import numpy as np
from numpy.typing import ArrayLike
import jinja2
import OpenGL.GL as gl
import glm
from enum import IntEnum
from dataclasses import dataclass
from ordered_set import OrderedSet

from topax.ops import Op, OpType, DType, RetType
from topax.sdfs import SDF, empty
from topax._utils import compile_shader, create_shader_program, load_texture, create_cube_vao

class ShaderMode(IntEnum):
    AMBIENT=0
    MOD_ONE=1
    MARCH_STEPS=2

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
    color = None

class ShaderGLSL:
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
    template = jinja2.Environment(loader=jinja2.PackageLoader('topax')).get_template('shader.glsl.j2')
    def __init__(self):
        self.program_id = None
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.uniform_locs = ShaderUniforms()
        self.sdfs = [] # array of SDF objects
        self.colors = [] # array of colors (RGB, 0-1)
        vs = compile_shader(ShaderGLSL.VIEW_CUBE_VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
        fs = compile_shader(ShaderGLSL.VIEW_CUBE_FRAGMENT_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
        self.view_cube_program_id = create_shader_program(vs, fs)
        self.view_cube_texture_id = load_texture("topax/resources/viewcube.png")
        self.view_cube_vao, self.view_cube_indices_len = create_cube_vao()

        sdfs = [empty()]
        colors = np.array([[0.0, 0.0, 0.0]])
        self.update_sdfs(sdfs, colors)

    def draw_view_cube(self, fb_height, camera_position, camera_up, vp_size=240):
        gl.glViewport(0, fb_height-vp_size, vp_size, vp_size)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        # Build projection + view
        projection = glm.ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 10.0)
        view = glm.lookAt(
            glm.normalize(glm.vec3(*camera_position)),
            glm.vec3(0,0,0),
            glm.vec3(*camera_up)
        )
        MVP = projection * view

        gl.glUseProgram(self.view_cube_program_id)
        loc = gl.glGetUniformLocation(self.view_cube_program_id, "MVP")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, np.array(MVP, dtype=np.float32).T)

        tex_loc = gl.glGetUniformLocation(self.view_cube_program_id, "cubeTexture")
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.view_cube_texture_id)
        gl.glUniform1i(tex_loc, 0)

        gl.glBindVertexArray(self.view_cube_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.view_cube_indices_len, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def draw(self, fb_width, fb_height, camera_position, looking_at, camera_up, fx, stop_epsilon, tmax, mode: ShaderMode, steps=1024):
        gl.glUseProgram(self.program_id)
        gl.glUniform2f(self.uniform_locs.i_resolution, fb_width, fb_height)
        gl.glUniform1ui(self.uniform_locs.max_steps, steps)
        gl.glUniform3f(self.uniform_locs.cam_pose, *camera_position)
        gl.glUniform3f(self.uniform_locs.looking_at, *looking_at)
        gl.glUniform3f(self.uniform_locs.cam_up, *camera_up)
        gl.glUniform1f(self.uniform_locs.fx, fx)
        gl.glUniform1f(self.uniform_locs.stop_epsilon, stop_epsilon)
        gl.glUniform1f(self.uniform_locs.tmax, tmax)
        gl.glUniform1ui(self.uniform_locs.mode, mode)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        self.draw_view_cube(fb_height, camera_position, camera_up)

    
    def update_sdfs(self, sdfs: list[SDF], colors: ArrayLike):
        colors = np.atleast_2d(colors).astype(np.float32)
        assert colors.shape[1] == 3
        assert colors.shape[0] == len(sdfs)
        if self.program_id: gl.glUseProgram(self.program_id)
        p = Op(OpType.CONST, ('p',), DType.vec3, value='p')
        optrees = [sdf(p) for sdf in sdfs]
        uniform_vars = {}
        # first determine if SDFs have changed structurally
        print([(new_sdf.hash(), old_sdf.hash()) for new_sdf, old_sdf in zip(sdfs, self.sdfs)])
        if len(sdfs) != len(self.sdfs) or any([new_sdf.hash() != old_sdf.hash() for new_sdf, old_sdf in zip(sdfs, self.sdfs)]):
            print("Recompiling shader...")
            maps = [ShaderGLSL.generate_shader_code(optree, f"sdf{i}") for i, optree in enumerate(optrees)]
            for m in maps:
                uniform_vars.update(m[1])
            code = ShaderGLSL.template.render(
                global_inputs=[ShaderGLSL.get_var_definition(g, k.rettype) for m in maps for k, g in m[1].items()],
                sdfs=[dict(name=f"sdf{i}", lines=m[0]) for i, m in enumerate(maps)],
            )
            # print(code)
            vs = compile_shader(ShaderGLSL.VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
            fs = compile_shader(code, gl.GL_FRAGMENT_SHADER)
            program_id = create_shader_program(vs, fs)
            gl.glUseProgram(program_id)
            if self.program_id is not None: gl.glDeleteProgram(self.program_id)
            self.program_id = program_id

            self.uniform_locs.i_resolution = gl.glGetUniformLocation(self.program_id, "_iResolution")
            self.uniform_locs.max_steps = gl.glGetUniformLocation(self.program_id, "_maxSteps")
            self.uniform_locs.cam_pose = gl.glGetUniformLocation(self.program_id, "_camPose")
            self.uniform_locs.looking_at = gl.glGetUniformLocation(self.program_id, "_lookingAt")
            self.uniform_locs.cam_up = gl.glGetUniformLocation(self.program_id, "_camUp")
            self.uniform_locs.fx = gl.glGetUniformLocation(self.program_id, "_fx")
            self.uniform_locs.stop_epsilon = gl.glGetUniformLocation(self.program_id, "_stopEpsilon")
            self.uniform_locs.tmax = gl.glGetUniformLocation(self.program_id, "_tmax")
            self.uniform_locs.color = gl.glGetUniformLocation(self.program_id, "sdf_colors")
            self.uniform_locs.mode = gl.glGetUniformLocation(self.program_id, "_mode")
        else:
            for i, optree in enumerate(optrees):
                uniform_vars.update(ShaderGLSL.get_global_vars(optree, f"sdf{i}"))

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ShaderGLSL.QUAD.nbytes, ShaderGLSL.QUAD, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glUniform3fv(self.uniform_locs.color, colors.shape[0], colors.flatten())

        for k, v in uniform_vars.items():
            location = gl.glGetUniformLocation(self.program_id, v)
            match k.rettype.dtype:
                case DType.float:
                    if k.rettype.length is not None: gl.glUniform1fv(location, k.rettype.length, np.atleast_1d(k.value).astype(np.float32))
                    else: gl.glUniform1f(location, float(k.value))
                case DType.vec2: gl.glUniform2f(location, *np.atleast_1d(k.value).astype(np.float32))
                case DType.vec3: gl.glUniform3f(location, *np.atleast_1d(k.value).astype(np.float32))
                case DType.vec4: gl.glUniform4f(location, *np.atleast_1d(k.value).astype(np.float32))
                case DType.mat2: gl.glUniformMatrix2fv(location, 1, gl.GL_FALSE, np.atleast_1d(k.value).astype(np.float32).flatten())
                case DType.mat3: gl.glUniformMatrix3fv(location, 1, gl.GL_FALSE, np.atleast_1d(k.value).astype(np.float32).flatten())
                case DType.mat4: gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, np.atleast_1d(k.value).astype(np.float32).flatten())
                case DType.int:
                    if k.rettype.length is not None: gl.glUniform1iv(location, k.rettype.length, np.atleast_1d(k.value).astype(np.int32))
                    else: gl.glUniform1i(location, int(k.value))
                case DType.ivec2: gl.glUniform2i(location, *np.atleast_1d(k.value).astype(np.int32))
                case DType.ivec3: gl.glUniform3i(location, *np.atleast_1d(k.value).astype(np.int32))
                case DType.ivec4: gl.glUniform4i(location, *np.atleast_1d(k.value).astype(np.int32))
                case _: raise TypeError(f"can't set uniform for type {k.rettype}")

        self.sdfs = sdfs
        self.colors = colors

    @staticmethod
    def traverse(op: Op) -> tuple[dict[Op, int], dict[Op, set[Op]], list[Op]]:
        """
        Traverse an op graph and build reversed graph and node input count, 
        as well as queue ready for tape generation.

        :param op: The operation graph to traverse

        :return in_count: map where key is a sub op and value is number of unique arguments
        :return consumer_nodes: map where key is sub op and value is all ops that consume this op
        :return consts: a list of constants (both static and global vars)
        """
        in_count = {}
        consumer_nodes = {}
        consts = []
        consts_seen = set()

        def _traverse(in_count: dict, consumer_nodes: dict, consts: list, consts_seen: set, op: Op):
            in_cnt = len(set(op.args))
            if op.opcode == OpType.CONST:
                if op not in consts_seen:
                    consts.append(op)
                    consts_seen.add(op)
            else:
                in_count[op] = in_cnt
                for o in op.args:
                    if o not in consumer_nodes: consumer_nodes[o] = set()
                    consumer_nodes[o].add(op)
                    _traverse(in_count, consumer_nodes, consts, consts_seen, o)

        _traverse(in_count, consumer_nodes, consts, consts_seen, op)
        return in_count, consumer_nodes, consts
    
    @staticmethod
    def make_tape(in_count: dict[Op, int], consumer_nodes: dict[Op, set[Op]], consts: list[Op]) -> list[Op]:
        """
        Perform a topological sort of the computation graph, generating
        a linear tape
        """
        if len(in_count.keys()) == 0: return consts
        queue = deque(consts)
        tape = []
        while len(queue) > 0:
            n = queue.popleft()
            if n.opcode != OpType.CONST:
                tape.append(n)
            else:
                assert n in consumer_nodes
                assert n in consts
            if n in consumer_nodes:
                for pn in consumer_nodes[n]:
                    in_count[pn] -= 1
                    if in_count[pn] == 0:
                        queue.append(pn)
            else:
                assert len(queue) == 0
        return tape
    
    @staticmethod
    def get_local_vars_ttl(tape: list[Op]) -> dict[Op, int]:
        vars_ttl = {}
        for i, l in enumerate(tape):
            vars_ttl[l] = i+1
            for arg in l.args:
                if arg.opcode == OpType.CONST: continue
                vars_ttl[arg] = i+1
        return vars_ttl
    
    @staticmethod
    def get_static_expression(const: Op) -> str:
        if isinstance(const.value, str): return const.value
        else:
            assert const.rettype.length is None
            match const.rettype.dtype:
                case DType.float: return f"{float(const.value)}"
                case DType.vec2: return f"vec2({float(const.value[0])},{float(const.value[1])})"
                case DType.vec3: return f"vec3({float(const.value[0])},{float(const.value[1])},{float(const.value[2])})"
                case DType.vec4: return f"vec4({float(const.value[0])},{float(const.value[1])},{float(const.value[2])},{float(const.value[3])})"
                case DType.mat2: return f"mat2({float(const.value[0,0])},{float(const.value[1,0])},{float(const.value[0,1])},{float(const.value[1,1])})"
                case DType.mat3: return f"mat3({float(const.value[0,0])},{float(const.value[1,0])},{float(const.value[2,0])},{float(const.value[0,1])},{float(const.value[1,1])},{float(const.value[2,1])},{float(const.value[0,2])},{float(const.value[1,2])},{float(const.value[2,2])})"
                case DType.mat4: return f"mat4({float(const.value[0,0])},{float(const.value[1,0])},{float(const.value[2,0])},{float(const.value[3,0])},{float(const.value[0,1])},{float(const.value[1,1])},{float(const.value[2,1])},{float(const.value[3,1])},{float(const.value[0,2])},{float(const.value[1,2])},{float(const.value[2,2])},{float(const.value[3,2])},{float(const.value[0,3])},{float(const.value[1,3])},{float(const.value[2,3])},{float(const.value[3,3])})"
                case DType.int: return f"{int(const.value)}"
                case DType.ivec2: return f"ivec2({int(const.value[0])},{int(const.value[1])})"
                case DType.ivec3: return f"ivec3({int(const.value[0])},{int(const.value[1])},{int(const.value[2])})"
                case DType.ivec4: return f"ivec4({int(const.value[0])},{int(const.value[1])},{int(const.value[2])},{int(const.value[3])})"
                case _: raise TypeError(f"expression for type {const.rettype.dtype} not yet supported")

    @staticmethod
    def get_var_definition(name: str, type: RetType) -> str:
        defin = ""
        match type.dtype:
            case DType.float: defin = f"float {name}"
            case DType.vec2: defin = f"vec2 {name}"
            case DType.vec3: defin = f"vec3 {name}"
            case DType.vec4: defin = f"vec4 {name}"
            case DType.mat2: defin = f"mat2 {name}"
            case DType.mat3: defin = f"mat3 {name}"
            case DType.mat4: defin = f"mat4 {name}"
            case DType.int: defin = f"int {name}"
            case DType.ivec2: defin = f"ivec2 {name}"
            case DType.ivec3: defin = f"ivec3 {name}"
            case DType.ivec4: defin = f"ivec4 {name}"
            case _: raise TypeError(f"var definition not supported for type {type} | name {name}")
        if type.length is not None:
            defin += f"[{int(type.length)}]"
        return defin

    @staticmethod
    def generate_shader_code(graph: Op, prefix: str) -> tuple[list[str], dict[Op, str]]:
        in_count, consumer_nodes, consts = ShaderGLSL.traverse(graph)
        tape = ShaderGLSL.make_tape(in_count, consumer_nodes, consts)
        vars_ttl = ShaderGLSL.get_local_vars_ttl(tape)

        lines = [] # the list of strings making up the shader code
        local_expressions = {} # shader string expressions representing ops, key is op value is string (can be local variable name or compute expression)
        local_var_ops = set() # will contain all ops that are stored to a local var
        local_vars_available = {} # local vars available for reuse, key is RetType value is string
        local_vars_total = 0
        global_vars = {c: f"{prefix}_sdfglobal_var{i}" for i, c in enumerate(consts) if c.sdf is not None}

        for ti, op in enumerate(tape):
            assert op not in local_expressions, "duplicate op expression found!"
            arg_expressions = [] # each element is a string for the corresponding op arg, either a variable name or a direct expression
            for ai, arg in enumerate(op.args):
                # if this is a constant static expression, just convert directly to expression
                if arg.opcode == OpType.CONST and arg.sdf is None: arg_expressions.append(ShaderGLSL.get_static_expression(arg))
                elif arg in local_expressions: # true if this arg is not a global var
                    arg_expressions.append(local_expressions[arg])
                    if vars_ttl[arg] == ti and arg not in op.args[ai+1:]: # if the last reference of this var is at this line
                        if arg in local_var_ops:
                            local_vars_available[arg.rettype].append(local_expressions[arg])
                            local_var_ops.remove(arg)
                        del local_expressions[arg]
                else:
                    try:
                        arg_expressions.append(global_vars[arg])
                    except KeyError as e:
                        print(ti, local_expressions)
                        raise e

            match op.opcode:
                case OpType.CONST: expression = f"({op.value})"
                case OpType.ADD: expression = f"({" + ".join(arg_expressions)})"
                case OpType.SUB: expression = f"({arg_expressions[0]} - {arg_expressions[1]})"
                case OpType.MUL: expression = f"({" * ".join(arg_expressions)})"
                case OpType.DIV: expression = f"({arg_expressions[0]} / {arg_expressions[1]})"
                case OpType.LEN: expression = f"length({arg_expressions[0]})"
                case OpType.DOT: expression = f"dot({arg_expressions[0]}, {arg_expressions[1]})"
                case OpType.YZX: expression = f"{arg_expressions[0]}.yzx"
                case OpType.XY: expression = f"{arg_expressions[0]}.xy"
                case OpType.XZ: expression = f"{arg_expressions[0]}.xz"
                case OpType.YZ: expression = f"{arg_expressions[0]}.yz"
                case OpType.X: expression = f"{arg_expressions[0]}.x"
                case OpType.Y: expression = f"{arg_expressions[0]}.y"
                case OpType.Z: expression = f"{arg_expressions[0]}.z"
                case OpType.SIN: expression = f"sin({arg_expressions[0]})"
                case OpType.COS: expression = f"cos({arg_expressions[0]})"
                case OpType.EXP2: expression = f"exp2({arg_expressions[0]})"
                case OpType.TAN: expression = f"tan({arg_expressions[0]})"
                case OpType.ASIN: expression = f"asin({", ".join(arg_expressions)})"
                case OpType.ACOS: expression = f"acos({", ".join(arg_expressions)})"
                case OpType.ATAN: expression = f"atan({", ".join(arg_expressions)})"
                case OpType.ABS: expression = f"abs({arg_expressions[0]})"
                case OpType.VEC2: expression = f"vec2({", ".join(arg_expressions)})"
                case OpType.VEC3: expression = f"vec3({", ".join(arg_expressions)})"
                case OpType.VEC4: expression = f"vec4({", ".join(arg_expressions)})"
                case OpType.MAT2: expression = f"mat2({", ".join(arg_expressions)})"
                case OpType.MAT3: expression = f"mat3({", ".join(arg_expressions)})"
                case OpType.MAT4: expression = f"mat4({", ".join(arg_expressions)})"
                case OpType.SUBIDX: expression = f"{arg_expressions[0]}[{arg_expressions[1]}]"
                case OpType.NEG: expression = f"-{arg_expressions[0]}"
                case OpType.MIN:
                    # TODO: change this to do binary tree min
                    # want to build min compute tree
                    # for number of args remaining, find largest power of 2 that is less than this number
                    expression = f"min({arg_expressions[0]}, {arg_expressions[1]})"
                    for i in range(2, len(arg_expressions)):
                        expression = f"min({expression}, {arg_expressions[i]})"
                case OpType.MAX:
                    # TODO: change this to do binary tree min
                    # want to build min compute tree
                    # for number of args remaining, find largest power of 2 that is less than this number
                    expression = f"max({arg_expressions[0]}, {arg_expressions[1]})"
                    for i in range(2, len(arg_expressions)):
                        expression = f"max({expression}, {arg_expressions[i]})"
                case _: raise TypeError(f"operation expression for opcode {op.opcode} not yet supported")

            # need to allocate a var if this op gets reused
            if op in consumer_nodes and len(consumer_nodes[op]) > 1:
                if op.rettype not in local_vars_available: local_vars_available[op.rettype] = deque()
                if len(local_vars_available[op.rettype]) > 0:
                    out_var_name = local_vars_available[op.rettype].popleft()
                    line = out_var_name
                else:
                    out_var_name = f"local_var{local_vars_total}"
                    local_vars_total += 1
                    line = ShaderGLSL.get_var_definition(out_var_name, op.rettype)
                lines.append(f"{line} = {expression};")
                local_expressions[op] = out_var_name
                local_var_ops.add(op)
            else:
                local_expressions[op] = expression

            if ti == len(tape)-1:
                # last expression, return it!
                lines.append(f"return {expression};")
        
        return lines, global_vars
    
    @staticmethod
    def get_global_vars(op: Op, prefix: str):
        consts = []
        consts_seen = set()

        def _traverse(consts: list, consts_seen: set, op: Op):
            if op.opcode == OpType.CONST:
                if op not in consts_seen:
                    consts.append(op)
                    consts_seen.add(op)
            else:
                for o in op.args:
                    _traverse(consts, consts_seen, o)

        _traverse(consts, consts_seen, op)

        return {c: f"{prefix}_sdfglobal_var{i}" for i, c in enumerate(consts) if c.sdf is not None}
