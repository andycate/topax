from dataclasses import dataclass, field
from typing import Any
import topax.types as types
from topax.types import DType, TypeEnum
# from topax.sdfs_new import sdf

@dataclass(frozen=True)
class OpBase:
    type: types.DType

    def __pos__(self): return self
    def __neg__(self): return neg(self)
    def __add__(self, rhs): return add(self, rhs)
    def __radd__(self, lhs): return add(lhs, self)
    def __sub__(self, rhs): return sub(self, rhs)
    def __rsub__(self, lhs): return sub(lhs, self)
    def __mul__(self, rhs): return mul(self, rhs)
    def __rmul__(self, lhs): return mul(lhs, self)
    def __truediv__(self, rhs): return div(self, rhs)
    def __rtruediv__(self, lhs): return div(lhs, self)

    def dot(self, rhs):
        return dot(self, rhs)

    @property
    def x(self): return x_swizzle(self)
    @property
    def y(self): return y_swizzle(self)
    @property
    def z(self): return z_swizzle(self)
    @property
    def xy(self): return xy_swizzle(self)
    @property
    def yz(self): return yz_swizzle(self)
    @property
    def xz(self): return xz_swizzle(self)
    @property
    def yzx(self): return yzx_swizzle(self)

    def grad(self, p):
        raise NotImplementedError()

    @staticmethod
    def _get_broadcasted_type(lhs, rhs):
        ranking = {
            DType(TypeEnum.int): 0,
            DType(TypeEnum.float): 1,
            DType(TypeEnum.ivec2): 2,
            DType(TypeEnum.vec2): 3,
            DType(TypeEnum.ivec3): 4,
            DType(TypeEnum.vec3): 5,
            DType(TypeEnum.ivec4): 6,
            DType(TypeEnum.vec4): 7,
            DType(TypeEnum.mat2): 8,
            DType(TypeEnum.mat3): 9,
            DType(TypeEnum.mat4): 10,
        }
        return lhs.type if ranking[lhs.type] > ranking[rhs.type] else rhs.type

@dataclass(frozen=True)
class const(OpBase):
    value: Any

    def __post_init__(self):
        if self.type is None: object.__setattr__(self, 'type', types.resolve_type(self.value))

    def grad(self, p):
        return const(self.type, self.type.zero())

@dataclass(frozen=True)
class param(OpBase):
    """class representing a tunable param (a uniform in a glsl shader)"""
    name: str
    value: Any = field(repr=False, hash=False)
    implicit: bool = False

    def __post_init__(self):
        if self.type is None: object.__setattr__(self, 'type', types.resolve_type(self.value))

    def _update_name(self, name: str):
        object.__setattr__(self, 'name', name)

    def grad(self, p):
        return const(self.type, self.type.one())

@dataclass(frozen=True)
class OpTree(OpBase):
    args: tuple[OpBase]

    def __post_init__(self):
        args = tuple([a if isinstance(a, OpBase) else const(None, a) for a in self.args])
        object.__setattr__(self, 'args', args)
    
    # def __repr__(self):
    #     return f'{self.__class__.__name__}({','.join([a.__repr__() for a in self.args])})'


class neg(OpTree):
    def __init__(self, lhs: OpBase):
        type = lhs.type
        super().__init__(type, args=(lhs,))

    def grad(self, p: param):
        return neg(self.args[0].grad(p))

class add(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

    def grad(self, p: param):
        return neg(self.args[0].grad(p))

class sub(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

class mul(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        assert rhs.type not in {DType(TypeEnum.mat2), DType(TypeEnum.mat3), DType(TypeEnum.mat4)}
        if lhs.type in {DType(TypeEnum.mat2), DType(TypeEnum.mat3), DType(TypeEnum.mat4)}:
            assert (lhs.type, rhs.type) in {(DType(TypeEnum.mat2), DType(TypeEnum.vec2)), (DType(TypeEnum.mat3), DType(TypeEnum.vec3)), (DType(TypeEnum.mat4), DType(TypeEnum.vec4))}
            type = rhs.type
        else:
            type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

class div(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

class sin(OpTree):
    def __init__(self, lhs: OpBase):
        super().__init__(lhs.type, args=(lhs,))

class cos(OpTree):
    def __init__(self, lhs: OpBase):
        super().__init__(lhs.type, args=(lhs,))

class length(OpTree):
    def __init__(self, lhs: OpBase):
        assert lhs.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}
        super().__init__(DType(TypeEnum.float), args=(lhs,))
    
class dot(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        assert lhs.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}
        assert lhs.type == rhs.type
        super().__init__(DType(TypeEnum.float), args=(lhs, rhs))

class min(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

class max(OpTree):
    def __init__(self, lhs: OpBase, rhs: OpBase):
        type = OpBase._get_broadcasted_type(lhs, rhs)
        super().__init__(type, args=(lhs, rhs))

class abs(OpTree):
    def __init__(self, lhs: OpBase):
        super().__init__(lhs.type, args=(lhs,))

class vec3(OpTree):
    def __init__(self, x: OpBase, y: OpBase, z: OpBase):
        super().__init__(DType(TypeEnum.vec3), args=(x, y, z))

class vec2(OpTree):
    def __init__(self, x: OpBase, y: OpBase):
        super().__init__(DType(TypeEnum.vec2), args=(x, y))

class x_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.float)
        else:
            type = DType(TypeEnum.int)
        super().__init__(type, args=(arg,))

class y_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.float)
        else:
            type = DType(TypeEnum.int)
        super().__init__(type, args=(arg,))

class z_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.float)
        else:
            type = DType(TypeEnum.int)
        super().__init__(type, args=(arg,))

class xy_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.vec2)
        else:
            type = DType(TypeEnum.ivec2)
        super().__init__(type, args=(arg,))

class yz_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.vec2)
        else:
            type = DType(TypeEnum.ivec2)
        super().__init__(type, args=(arg,))

class xz_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.vec2)
        else:
            type = DType(TypeEnum.ivec2)
        super().__init__(type, args=(arg,))

class yzx_swizzle(OpTree):
    def __init__(self, arg: OpBase):
        if arg.type in {DType(TypeEnum.vec2), DType(TypeEnum.vec3), DType(TypeEnum.vec4)}:
            type = DType(TypeEnum.vec3)
        else:
            type = DType(TypeEnum.ivec3)
        super().__init__(type, args=(arg,))

    