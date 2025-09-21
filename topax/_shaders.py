from typing import Any
from collections import deque
import numpy as np
from numpy.typing import ArrayLike
import jinja2
import OpenGL.GL as gl

from topax.ops import Op, OpType, DType, RetType
from topax.sdfs import SDF, empty
from topax._utils import compile_shader

class ShaderGLSL:
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
    template = jinja2.Environment(loader=jinja2.PackageLoader('topax')).get_template('shader.glsl.j2')
    def __init__(self):
        self.program_id = None
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.loc_i_resolution = None
        self.loc_max_steps = None
        self.loc_cam_pose = None
        self.loc_looking_at = None
        self.loc_cam_up = None
        self.loc_fx = None
        self.loc_stop_epsilon = None
        self.loc_tmax = None
        self.loc_color = None
        self.sdfs = []
        self.colors = []
        sdfs = [empty()]
        colors = np.array([[0.0, 0.0, 0.0]])
        self.update_sdfs(sdfs, colors)


    def draw(self, fb_width, fb_height, camera_position, looking_at, camera_up, fx):
        gl.glUniform2f(self.loc_i_resolution, fb_width, fb_height)
        gl.glUniform1ui(self.loc_max_steps, 1024)
        gl.glUniform3f(self.loc_cam_pose, *camera_position)
        gl.glUniform3f(self.loc_looking_at, *looking_at)
        gl.glUniform3f(self.loc_cam_up, *camera_up)
        gl.glUniform1f(self.loc_fx, fx)
        gl.glUniform1f(self.loc_stop_epsilon, 0.00001)
        gl.glUniform1f(self.loc_tmax, 1000.0)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    
    def update_sdfs(self, sdfs: list[SDF], colors: ArrayLike):
        colors = np.atleast_2d(colors).astype(np.float32)
        assert colors.shape[1] == 3
        assert colors.shape[0] == len(sdfs)
        if self.program_id: gl.glUseProgram(self.program_id)
        p = Op(OpType.CONST, ('p',), DType.vec3, value='p')
        optrees = [sdf(p) for sdf in sdfs]
        uniform_vars = {}
        # first determine if SDFs have changed structurally
        if len(sdfs) != len(self.sdfs) or any([new_sdf.hash() != old_sdf.hash() for new_sdf, old_sdf in zip(sdfs, self.sdfs)]):
            maps = [ShaderGLSL.generate_shader_code(optree, f"sdf{i}") for i, optree in enumerate(optrees)]
            for m in maps:
                uniform_vars.update(m[1])
            code = ShaderGLSL.template.render(
                global_inputs=[ShaderGLSL.get_var_definition(g, k.rettype) for m in maps for k, g in m[1].items()],
                sdfs=[dict(name=f"sdf{i}", lines=m[0]) for i, m in enumerate(maps)],
            )
            vs = compile_shader(ShaderGLSL.VERTEX_SHADER_SOURCE, gl.GL_VERTEX_SHADER)
            fs = compile_shader(code, gl.GL_FRAGMENT_SHADER)
            program_id = gl.glCreateProgram()
            gl.glAttachShader(program_id, vs)
            gl.glAttachShader(program_id, fs)
            gl.glLinkProgram(program_id)
            if not gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS):
                raise RuntimeError(gl.glGetProgramInfoLog(program_id).decode())
            gl.glDeleteShader(vs)
            gl.glDeleteShader(fs)
            gl.glUseProgram(program_id)
            if self.program_id is not None: gl.glDeleteProgram(self.program_id)
            self.program_id = program_id

            self.loc_i_resolution = gl.glGetUniformLocation(self.program_id, "_iResolution")
            self.loc_max_steps = gl.glGetUniformLocation(self.program_id, "_maxSteps")
            self.loc_cam_pose = gl.glGetUniformLocation(self.program_id, "_camPose")
            self.loc_looking_at = gl.glGetUniformLocation(self.program_id, "_lookingAt")
            self.loc_cam_up = gl.glGetUniformLocation(self.program_id, "_camUp")
            self.loc_fx = gl.glGetUniformLocation(self.program_id, "_fx")
            self.loc_stop_epsilon = gl.glGetUniformLocation(self.program_id, "_stopEpsilon")
            self.loc_tmax = gl.glGetUniformLocation(self.program_id, "_tmax")
            self.loc_color = gl.glGetUniformLocation(self.program_id, "sdf_colors")
        else:
            for i, optree in enumerate(optrees):
                uniform_vars.update(ShaderGLSL.get_global_vars(optree, f"sdf{i}"))

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ShaderGLSL.QUAD.nbytes, ShaderGLSL.QUAD, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glUniform3fv(self.loc_color, colors.shape[0], colors.flatten())

        for k, v in uniform_vars.items():
            location = gl.glGetUniformLocation(self.program_id, v)
            match k.rettype:
                case RetType(DType.float): gl.glUniform1f(location, float(k.value))
                case RetType(DType.vec2): gl.glUniform2f(location, *np.atleast_1d(k.value).astype(np.float32))
                case RetType(DType.vec3): gl.glUniform3f(location, *np.atleast_1d(k.value).astype(np.float32))
                case RetType(DType.vec4): gl.glUniform4f(location, *np.atleast_1d(k.value).astype(np.float32))
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
            vars_ttl[l] = i
            for arg in l.args:
                if arg.opcode == OpType.CONST: continue
                vars_ttl[arg] = i
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
                case _: raise TypeError(f"expression for type {const.rettype.dtype} not yet supported")

    @staticmethod
    def get_var_definition(name: str, type: RetType) -> str:
        assert type.length is None
        match type.dtype:
            case DType.float: return f"float {name}"
            case DType.vec2: return f"vec2 {name}"
            case DType.vec3: return f"vec3 {name}"
            case DType.vec4: return f"vec4 {name}"
            case _: raise TypeError(f"var definition not supported for type {type}")

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
            for arg in op.args:
                # if this is a constant static expression, just convert directly to expression
                if arg.opcode == OpType.CONST and arg.sdf is None: arg_expressions.append(ShaderGLSL.get_static_expression(arg))
                elif arg in local_expressions: # true if this arg is not a global var
                    arg_expressions.append(local_expressions[arg])
                    if vars_ttl[arg] == ti: # if the last reference of this var is at this line
                        if arg in local_var_ops:
                            local_vars_available[arg.rettype].append(local_expressions[arg])
                            local_var_ops.remove(arg)
                        del local_expressions[arg]
                else:
                    try:
                        arg_expressions.append(global_vars[arg])
                    except KeyError as e:
                        # print(local_expressions)
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
                case OpType.X: expression = f"{arg_expressions[0]}.x"
                case OpType.Y: expression = f"{arg_expressions[0]}.y"
                case OpType.Z: expression = f"{arg_expressions[0]}.z"
                case OpType.SIN: expression = f"sin({arg_expressions[0]})"
                case OpType.COS: expression = f"cos({arg_expressions[0]})"
                case OpType.ABS: expression = f"abs({arg_expressions[0]})"
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
