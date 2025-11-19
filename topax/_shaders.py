from collections import deque
from ordered_set import OrderedSet
from enum import IntEnum
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike
import jinja2
import OpenGL.GL as gl
from pyglm import glm

import topax.ops as ops
from topax.sdfs import SDF, sphere
from topax.types import DType, BaseType
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
    template_3d = jinja2.Environment(loader=jinja2.PackageLoader('topax')).get_template('shader_3d.glsl.j2')
    template_2d = jinja2.Environment(loader=jinja2.PackageLoader('topax')).get_template('shader_2d.glsl.j2')
    def __init__(self, print_code=False):
        self.print_code = print_code
        self.program_id = None
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.uniform_locs = ShaderUniforms()
        vs = compile_shader(ShaderGLSL.VIEW_CUBE_VERTEX_SHADER_SRC, gl.GL_VERTEX_SHADER)
        fs = compile_shader(ShaderGLSL.VIEW_CUBE_FRAGMENT_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
        self.view_cube_program_id = create_shader_program(vs, fs)
        self.view_cube_texture_id = load_texture("topax/resources/viewcube.png")
        self.view_cube_vao, self.view_cube_indices_len = create_cube_vao()

        self.sdf_shaders = []
        self.colors = []
        self.global_uniforms = []
        self.update_sdfs([sphere(0.0)], [[0.0, 0.0, 0.0]])

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
        assert all([s.is_2d for s in sdfs]) or all([not s.is_2d for s in sdfs])
        self.is_2d_mode = all([s.is_2d for s in sdfs])
        if self.program_id: gl.glUseProgram(self.program_id)

        recompile = False
        if len(sdfs) != len(self.sdf_shaders):
            recompile = True
            self.sdf_shaders = [ShaderSDF(sdf, f"sdf{i}") for i, sdf in enumerate(sdfs)]
        else:
            for i, sdf_shader in enumerate(self.sdf_shaders):
                needs_update = sdf_shader.update_sdf(sdfs[i])
                if needs_update:
                    self.sdf_shaders[i] = ShaderSDF(sdfs[i], self.sdf_shaders[i].prefix)
                    recompile = True
        
        self.colors = colors
        self.global_uniforms = []
        for ss in self.sdf_shaders:
            for ep in ss.get_explicit_params():
                if ep not in self.global_uniforms: self.global_uniforms.append(ep)
        for ss in self.sdf_shaders: self.global_uniforms.extend(ss.get_prefixed_implicit_params())

        if recompile:
            print("Recompiling shader...")
            if self.is_2d_mode:
                code = ShaderGLSL.template_2d.render(
                    global_uniforms=self.global_uniforms,
                    sdfs=self.sdf_shaders,
                )
            else:
                code = ShaderGLSL.template_3d.render(
                    global_uniforms=self.global_uniforms,
                    sdfs=self.sdf_shaders,
                )
            if self.print_code: print(code)
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

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, ShaderGLSL.QUAD.nbytes, ShaderGLSL.QUAD, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glUniform3fv(self.uniform_locs.color, self.colors.shape[0], self.colors.flatten())

        for u in self.global_uniforms:
            location = gl.glGetUniformLocation(self.program_id, u.name)
            match u.dtype.base:
                case BaseType.float:
                    if u.dtype.length is not None: gl.glUniform1fv(location, u.dtype.length, np.atleast_1d(u.value).astype(np.float32))
                    else: gl.glUniform1f(location, float(u.value))
                case BaseType.vec2: gl.glUniform2f(location, *np.atleast_1d(u.value).astype(np.float32))
                case BaseType.vec3: gl.glUniform3f(location, *np.atleast_1d(u.value).astype(np.float32))
                case BaseType.vec4: gl.glUniform4f(location, *np.atleast_1d(u.value).astype(np.float32))
                case BaseType.mat2: gl.glUniformMatrix2fv(location, 1, gl.GL_FALSE, np.atleast_1d(u.value).astype(np.float32).flatten())
                case BaseType.mat3: gl.glUniformMatrix3fv(location, 1, gl.GL_FALSE, np.atleast_1d(u.value).astype(np.float32).flatten())
                case BaseType.mat4: gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, np.atleast_1d(u.value).astype(np.float32).flatten())
                case BaseType.int:
                    if u.dtype.length is not None: gl.glUniform1iv(location, u.dtype.length, np.atleast_1d(u.value).astype(np.int32))
                    else: gl.glUniform1i(location, int(u.value))
                case _: raise TypeError(f"can't set uniform for type {u.dtype}")
        return OrderedSet([ep for ss in self.sdf_shaders for ep in ss.get_explicit_params()])
    
    def update_explicit_param(self, name, value):
        gl.glUseProgram(self.program_id)
        location = gl.glGetUniformLocation(self.program_id, name)
        gl.glUniform1f(location, float(value))


class ShaderSDF:
    """Class to encapsulate all the shader code generated for an SDF"""
    def __init__(self, sdf: SDF, prefix=''):
        self.sdf = sdf
        self.prefix = prefix
        
        if self.sdf.is_2d:
            p = ops.param(DType(BaseType.vec2), '_p', None)
        else:
            p = ops.param(DType(BaseType.vec3), '_p', None)
        self.tree = self.sdf(p)
        in_count, consumer_nodes, leaves = ShaderSDF._traverse(self.tree)
        tape = ShaderSDF._make_tape(in_count, consumer_nodes, leaves)
        ttl = ShaderSDF._local_vars_ttl(tape)

        self.map_func = self.generate_map_func(tape, ttl)

        self.grad_tree = self.tree.grad(p)
        in_count, consumer_nodes, leaves = ShaderSDF._traverse(self.grad_tree)
        tape = ShaderSDF._make_tape(in_count, consumer_nodes, leaves)
        ttl = ShaderSDF._local_vars_ttl(tape)

        self.map_grad_func = self.generate_map_func(tape, ttl)

    def update_sdf(self, new_sdf):
        if new_sdf.is_2d:
            p = ops.param(DType(BaseType.vec2), '_p', None)
        else:
            p = ops.param(DType(BaseType.vec3), '_p', None)
        tree = new_sdf(p)
        if hash(tree) != hash(self.tree):
            return True
        else:
            self.sdf = new_sdf
            return False
        
    def get_explicit_params(self):
        return self.sdf._explicit_params
    
    def get_prefixed_implicit_params(self):
        return [replace(p, name=self.prefix+p.name) for p in self.sdf._implicit_params]

    @staticmethod
    def _traverse(op: ops.OpTree):
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
        leaves = OrderedSet()

        def _traverse_inner(in_count: dict, consumer_nodes: dict, leaves: OrderedSet[ops.OpTree], op: ops.OpTree):
            if isinstance(op, ops.const) or isinstance(op, ops.param): return
            in_count[op] = len(set([a for a in op.args if isinstance(a, ops.OpTree)]))
            is_leaf = True
            for o in op.args:
                if isinstance(o, ops.OpTree):
                    is_leaf = False
                    if o not in consumer_nodes: consumer_nodes[o] = OrderedSet()
                    consumer_nodes[o].add(op)
                    _traverse_inner(in_count, consumer_nodes, leaves, o)
            if is_leaf:
                leaves.add(op)

        _traverse_inner(in_count, consumer_nodes, leaves, op)
        return in_count, consumer_nodes, leaves

    @staticmethod
    def _make_tape(
        in_count: dict[ops.OpTree, int], 
        consumer_nodes: dict[ops.OpBase, OrderedSet[ops.OpTree]], 
        leaves: OrderedSet[ops.OpTree]
    ) -> list[ops.OpTree]:
        """
        Perform a topological sort of the computation graph, generating
        a linear tape
        """
        queue = deque(leaves)
        tape = []
        while len(queue) > 0:
            n = queue.popleft()
            tape.append(n)
            if n in consumer_nodes:
                for pn in consumer_nodes[n]:
                    in_count[pn] -= 1
                    if in_count[pn] == 0:
                        queue.append(pn)
            else:
                assert len(queue) == 0
        return tape
    
    @staticmethod
    def _local_vars_ttl(tape: list[ops.OpTree]) -> dict[ops.OpTree, int]:
        vars_ttl = {}
        for i, l in enumerate(tape):
            for arg in l.args:
                if not isinstance(arg, ops.OpTree): continue
                vars_ttl[arg] = i
        return vars_ttl
    
    @staticmethod
    def _get_shader_const(c: ops.const):
        if hasattr(c.value, '__iter__'):
            val_str = ",".join([f"{v}" for v in c.value])
        else:
            val_str = f"{c.value}"
        match c.dtype.base:
            case BaseType.float: return f"{float(c.value)}"
            case BaseType.int: return f"{int(c.value)}"
            case BaseType.vec2: return f"vec2({val_str})"
            case BaseType.vec3: return f"vec3({val_str})"
            case BaseType.vec4: return f"vec4({val_str})"
            case _: pass
        raise NotImplementedError()
    
    @staticmethod
    def _get_shader_expression(op: ops.OpTree, var_map: dict, prefix: str):
        _var_map = var_map.copy()
        _var_map.update({a: (f"{prefix}{a.name}" if a.implicit else a.name) if isinstance(a, ops.param) else ShaderSDF._get_shader_const(a) for a in op.args if not isinstance(a, ops.OpTree)})
        args = [_var_map[a] for a in op.args]
        match op.optype:
            case ops.OpType.NEG: return f"-{args[0]}"
            case ops.OpType.ADD: return f"{args[0]} + {args[1]}"
            case ops.OpType.SUB: return f"{args[0]} - {args[1]}"
            case ops.OpType.MUL: return f"{args[0]} * {args[1]}"
            case ops.OpType.DIV: return f"{args[0]} / {args[1]}"
            case ops.OpType.LEN: return f"length({args[0]})"
            case ops.OpType.NORM: return f"normalize({args[0]})"
            case ops.OpType.DOT: return f"dot({args[0]},{args[1]})"
            case ops.OpType.CROSS: return f"cross({args[0]},{args[1]})"
            case ops.OpType.SQUARE: return f"square({args[0]})"
            case ops.OpType.SQRT: return f"sqrt({args[0]})"
            case ops.OpType.POW: return f"pow({args[0]},{args[1]})"
            case ops.OpType.SIN: return f"sin({args[0]})"
            case ops.OpType.COS: return f"cos({args[0]})"
            case ops.OpType.TAN: return f"tan({args[0]})"
            case ops.OpType.ASIN: return f"asin({args[0]})"
            case ops.OpType.ACOS: return f"acos({args[0]})"
            case ops.OpType.ATAN: return f"atan({args[0]},{args[1]})"
            case ops.OpType.MIN: return f"min({args[0]},{args[1]})"
            case ops.OpType.MAX: return f"max({args[0]},{args[1]})"
            case ops.OpType.ABS: return f"abs({args[0]})"
            case ops.OpType.EXP: return f"exp({args[0]})"
            case ops.OpType.EXP2: return f"exp2({args[0]})"
            case ops.OpType.LOG: return f"log({args[0]})"
            case ops.OpType.LOG2: return f"log2({args[0]})"
            case ops.OpType.MOD: return f"mod({args[0]},{args[1]})"
            case ops.OpType.CLAMP: return f"clamp({args[0]},{args[1]},{args[2]})"
            case ops.OpType.ROUND: return f"round({args[0]})"
            case ops.OpType.FLOOR: return f"floor({args[0]})"
            case ops.OpType.CEIL: return f"ceil({args[0]})"
            case ops.OpType.SIGN: return f"sign({args[0]})"
            case ops.OpType.VEC2: return f"vec2({",".join(args)})"
            case ops.OpType.VEC3: return f"vec3({",".join(args)})"
            case ops.OpType.VEC4: return f"vec4({",".join(args)})"
            case ops.OpType.MAT2: return f"mat2({",".join(args)})"
            case ops.OpType.MAT3: return f"mat3({",".join(args)})"
            case ops.OpType.MAT4: return f"mat4({",".join(args)})"
            case ops.OpType.X: return f"({args[0]}).x"
            case ops.OpType.Y: return f"({args[0]}).y"
            case ops.OpType.Z: return f"({args[0]}).z"
            case ops.OpType.XY: return f"({args[0]}).xy"
            case ops.OpType.YZ: return f"({args[0]}).yz"
            case ops.OpType.XZ: return f"({args[0]}).xz"
            case ops.OpType.YZX: return f"({args[0]}).yzx"
            case ops.OpType.TERNARY: return f"{args[0]} ? {args[1]} : {args[2]}"
            case ops.OpType.LT: return f"{args[0]} < {args[1]}"
            case ops.OpType.GT: return f"{args[0]} > {args[1]}"
            case _: pass
        raise NotImplementedError(f"{op.optype}")
    
    def generate_map_func(self, tape, ttl):
        lines = []
        var_cnt = 0
        var_map = {} # key is optree, value is name of variable
        var_pool = {} # key is DType, value is deque of variable names
        var_ttl = {} # key is var name, value is (iteration at which it is released, dtype)
        for i, op in enumerate(tape):
            # get local var that we can use
            if op.dtype not in var_pool: var_pool[op.dtype] = deque()
            for k in var_ttl:
                if var_ttl[k][0] == i:
                    var_pool[var_ttl[k][1]].appendleft(k)
            if len(var_pool[op.dtype]) == 0:
                local_var = f'_local_var_{var_cnt}'
                var_cnt += 1
                new_var = True
            else:
                local_var = var_pool[op.dtype].pop()
                new_var = False
            if op not in ttl:
                assert i == len(tape) - 1
                lines.append(f'return {ShaderSDF._get_shader_expression(op, var_map, self.prefix)};')
            else:
                var_ttl[local_var] = (ttl[op], op.dtype)
                # add line
                lines.append(f'{op.dtype.get_shader_declaration(local_var) if new_var else local_var} = {ShaderSDF._get_shader_expression(op, var_map, self.prefix)};')
                # end add line

            var_map[op] = local_var
        return '\n'.join(lines)

