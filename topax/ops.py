from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from functools import wraps
from typing import Callable
import numpy as np
import topax.types as types
from topax.types import DType, BaseType

class OpType(Enum):
    NEG = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    LEN = 5
    NORM = 6
    DOT = 7
    SQUARE = 8
    SQRT = 9
    POW = 10
    SIN = 11
    COS = 12
    TAN = 13
    ASIN = 14
    ACOS = 15
    ATAN = 16
    MIN = 17
    MAX = 18
    ABS = 19
    SUBIDX = 20
    EXP = 21
    EXP2 = 22
    LOG = 23
    LOG2 = 24
    MOD = 25
    CLAMP = 26
    ROUND = 27
    SIGN = 28
    VEC2 = 29
    VEC3 = 30
    VEC4 = 31
    MAT2 = 32
    MAT3 = 33
    MAT4 = 34
    X = 35
    Y = 36
    Z = 37
    W = 38
    XY = 39
    XZ = 40
    YZ = 41
    YZX = 42
    ZXY = 43
    TERNARY = 44
    LT = 45
    GT = 46

@dataclass(frozen=True)
class OpBase:
    dtype: DType # output data type of this op

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
    def __lt__(self, other): return lt(self, other)
    def __gt__(self, other): return gt(self, other)

    def dot(self, rhs):
        return dot(self, rhs)

    @property
    def x(self): return swizzle_x(self)
    @property
    def y(self): return swizzle_y(self)
    @property
    def z(self): return swizzle_z(self)
    @property
    def w(self): return swizzle_w(self)
    @property
    def xy(self): return swizzle_xy(self)
    @property
    def yz(self): return swizzle_yz(self)
    @property
    def xz(self): return swizzle_xz(self)
    @property
    def yzx(self): return swizzle_yzx(self)

    def grad(self, p):
        raise NotImplementedError()
        

@dataclass(frozen=True)
class const(OpBase):
    value: Any

    def __post_init__(self):
        if self.dtype is None: object.__setattr__(self, 'dtype', types.resolve_dtype(self.value))
        if isinstance(self.value, np.ndarray): object.__setattr__(self, 'value', tuple(self.value.flatten()))

    def grad(self, p):
        assert self.dtype.length is None
        assert self.dtype.base in {BaseType.float, BaseType.vec2, BaseType.vec3, BaseType.vec4}
        return const(self.dtype, 0.0)

@dataclass(frozen=True)
class param(OpBase):
    """class representing a tunable param (a uniform in a glsl shader)"""
    name: str
    value: Any = field(repr=False, hash=False)
    implicit: bool = False

    def __post_init__(self):
        if self.dtype is None: object.__setattr__(self, 'dtype', types.resolve_dtype(self.value))

    def _update_name(self, name: str):
        object.__setattr__(self, 'name', name)

    def grad(self, p):
        assert self.dtype.length is None
        assert self.dtype.base in {BaseType.float, BaseType.vec2, BaseType.vec3, BaseType.vec4}
        if self == p: return const(self.dtype, 1.0)
        else: return const(self.dtype, 0.0)

@dataclass(frozen=True)
class OpTree(OpBase):
    optype: OpType
    args: tuple[OpBase, ...]

    def __post_init__(self):
        args = tuple([a if isinstance(a, OpBase) else const(None, a) for a in self.args])
        object.__setattr__(self, 'args', args)

    @staticmethod
    def _get_broadcasted_type(lhs: DType, rhs: DType):
        assert isinstance(lhs, DType)
        assert isinstance(rhs, DType)
        if lhs.length != None or rhs.length != None:
            raise NotImplementedError("Array broadcasting not supported yet")
        if lhs == rhs: return lhs
        if lhs == DType(BaseType.float): return rhs
        if rhs == DType(BaseType.float): return lhs
        raise NotImplementedError(f"Type broadcasting between {lhs} and {rhs} not supported yet")
    
    def grad(self, p):
        match self.optype:
            case OpType.NEG: return -self.args[0].grad(p)
            case OpType.ADD: return self.args[0].grad(p) + self.args[1].grad(p)
            case OpType.SUB: return self.args[0].grad(p) - self.args[1].grad(p)
            case OpType.MUL: return (self.args[0] * self.args[1].grad(p)) + (self.args[0].grad(p) * self.args[1])
            case OpType.LEN:
                arg = self.args[0]
                assert arg.dtype.length is None
                assert arg.dtype.base in {BaseType.vec2, BaseType.vec3, BaseType.vec4}
                match arg.dtype.base:
                    case BaseType.vec2: return vec2(arg.grad(p).x * arg.x / self, arg.grad(p).y * arg.y / self)
                    case BaseType.vec3: return vec3(arg.grad(p).x * arg.x / self, arg.grad(p).y * arg.y / self, arg.grad(p).z * arg.z / self)
                    case BaseType.vec4: return vec4(arg.grad(p).x * arg.x / self, arg.grad(p).y * arg.y / self, arg.grad(p).z * arg.z / self, arg.grad(p).w * arg.w / self)
                    case _: pass
            case OpType.MIN:
                assert self.args[0].dtype.length is None and self.args[1].dtype.length is None
                if self.args[0].dtype.base == BaseType.float:
                    if self.args[1].dtype.base == BaseType.float:
                        return ternary(self.args[0] < self.args[1], self.args[0].grad(p), self.args[1].grad(p))
                    elif self.args[1].dtype.base == BaseType.vec2:
                        return vec2(ternary(self.args[0] < self.args[1].x, self.args[0].grad(p), self.args[1].x.grad(p)), ternary(self.args[0] < self.args[1].y, self.args[0].grad(p), self.args[1].y.grad(p)))
                    elif self.args[1].dtype.base == BaseType.vec3:
                        return vec3(ternary(self.args[0] < self.args[1].x, self.args[0].grad(p), self.args[1].x.grad(p)), ternary(self.args[0] < self.args[1].y, self.args[0].grad(p), self.args[1].y.grad(p)), ternary(self.args[0] < self.args[1].z, self.args[0].grad(p), self.args[1].z.grad(p)))
                elif self.args[1].dtype.base == BaseType.float:
                    if self.args[0].dtype.base == BaseType.vec2:
                        return vec2(ternary(self.args[0].x < self.args[1], self.args[0].x.grad(p), self.args[1].grad(p)), ternary(self.args[0].y < self.args[1], self.args[0].y.grad(p), self.args[1].grad(p)))
                    elif self.args[0].dtype.base == BaseType.vec3:
                        return vec3(ternary(self.args[0].x < self.args[1], self.args[0].x.grad(p), self.args[1].grad(p)), ternary(self.args[0].y < self.args[1], self.args[0].y.grad(p), self.args[1].grad(p)), ternary(self.args[0].z < self.args[1], self.args[0].z.grad(p), self.args[1].grad(p)))
                raise NotImplementedError(f"MIN gradient for args {self.args[0].dtype} and {self.args[1].dtype} not supported!")
            case OpType.MAX:
                assert self.args[0].dtype.length is None and self.args[1].dtype.length is None
                if self.args[0].dtype.base == BaseType.float:
                    if self.args[1].dtype.base == BaseType.float:
                        return ternary(self.args[0] > self.args[1], self.args[0].grad(p), self.args[1].grad(p))
                    elif self.args[1].dtype.base == BaseType.vec2:
                        return vec2(ternary(self.args[0] > self.args[1].x, self.args[0].grad(p), self.args[1].x.grad(p)), ternary(self.args[0] > self.args[1].y, self.args[0].grad(p), self.args[1].y.grad(p)))
                    elif self.args[1].dtype.base == BaseType.vec3:
                        return vec3(ternary(self.args[0] > self.args[1].x, self.args[0].grad(p), self.args[1].x.grad(p)), ternary(self.args[0] > self.args[1].y, self.args[0].grad(p), self.args[1].y.grad(p)), ternary(self.args[0] > self.args[1].z, self.args[0].grad(p), self.args[1].z.grad(p)))
                elif self.args[1].dtype.base == BaseType.float:
                    if self.args[0].dtype.base == BaseType.vec2:
                        return vec2(ternary(self.args[0].x > self.args[1], self.args[0].x.grad(p), self.args[1].grad(p)), ternary(self.args[0].y > self.args[1], self.args[0].y.grad(p), self.args[1].grad(p)))
                    elif self.args[0].dtype.base == BaseType.vec3:
                        return vec3(ternary(self.args[0].x > self.args[1], self.args[0].x.grad(p), self.args[1].grad(p)), ternary(self.args[0].y > self.args[1], self.args[0].y.grad(p), self.args[1].grad(p)), ternary(self.args[0].z > self.args[1], self.args[0].z.grad(p), self.args[1].grad(p)))
                raise NotImplementedError(f"MIN gradient for args {self.args[0].dtype} and {self.args[1].dtype} not supported!")
            case OpType.ABS:
                assert self.args[0].dtype.length == None
                # if self.args[0].dtype.base == BaseType.float: return self.args[0].grad(p) * sign(self.args[0])
                # elif self.args[0].dtype.base == BaseType.vec2: return vec2(self.args[0].grad(p).x * sign(self.args[0].x), self.args[0].grad(p).y * sign(self.args[0].y))
                # elif self.args[0].dtype.base == BaseType.vec3: return vec3(self.args[0].grad(p).x * sign(self.args[0].x), self.args[0].grad(p).y * sign(self.args[0].y))
                return self.args[0].grad(p) * sign(self.args[0])
            case OpType.VEC2: return vec2(*[a.grad(p) for a in self.args])
            case OpType.VEC3: return vec3(*[a.gr ad(p) for a in self.args])
            case OpType.X: return self.args[0].grad(p).x
            case OpType.Y: return self.args[0].grad(p).y
            case OpType.Z: return self.args[0].grad(p).z
            case OpType.XY: return self.args[0].grad(p).xy
            case OpType.YZ: return self.args[0].grad(p).yz
            case OpType.XZ: return self.args[0].grad(p).xz
            case _: pass
        raise NotImplementedError(f"Grad for operation {self.optype} not supported yet")


def wrap_const(func: Callable):
    @wraps(func)
    def wrapped(*args):
        args_wrapped = []
        for a in args:
            if not isinstance(a, OpBase):
                a = const(None, a)
            args_wrapped.append(a)
        return func(*args_wrapped)
    return wrapped

@wrap_const
def neg(lhs):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.NEG, args=(lhs,))

@wrap_const
def add(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.ADD, args=(lhs, rhs))

@wrap_const
def sub(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.SUB, args=(lhs, rhs))

@wrap_const
def mul(lhs: OpBase, rhs: OpBase):
    if lhs.dtype.base in {BaseType.mat2, BaseType.mat3, BaseType.mat4} and rhs.dtype in {BaseType.vec2, BaseType.vec3, BaseType.vec4}:
        # this is a matrix multiplication
        assert lhs.dtype.length is None and rhs.dtype.length is None
        dtype = rhs.dtype
    else:
        dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.MUL, args=(lhs, rhs))

@wrap_const
def div(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.DIV, args=(lhs, rhs))

@wrap_const
def length(lhs: OpBase):
    dtype = DType(BaseType.float)
    return OpTree(dtype, OpType.LEN, args=(lhs,))

@wrap_const
def norm(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.NORM, args=(lhs,))

@wrap_const
def dot(lhs: OpBase, rhs: OpBase):
    dtype = DType(BaseType.float)
    return OpTree(dtype, OpType.LEN, args=(lhs, rhs))

@wrap_const
def square(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.SQUARE, args=(lhs,))

@wrap_const
def sqrt(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.SQRT, args=(lhs,))

@wrap_const
def pow(lhs: OpBase, rhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.POW, args=(lhs, rhs))

@wrap_const
def sin(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.SIN, args=(lhs,))

@wrap_const
def cos(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.COS, args=(lhs,))

@wrap_const
def tan(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.TAN, args=(lhs,))

@wrap_const
def asin(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.ASIN, args=(lhs,))

@wrap_const
def acos(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.ACOS, args=(lhs,))

@wrap_const
def atan(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.ATAN, args=(lhs, rhs))

@wrap_const
def min(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.MIN, args=(lhs, rhs))

@wrap_const
def max(lhs: OpBase, rhs: OpBase):
    dtype = OpTree._get_broadcasted_type(lhs.dtype, rhs.dtype)
    return OpTree(dtype, OpType.MAX, args=(lhs, rhs))

@wrap_const
def abs(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.ABS, args=(lhs,))

@wrap_const
def exp(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.EXP, args=(lhs,))

@wrap_const
def exp2(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.EXP2, args=(lhs,))

@wrap_const
def log(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.LOG, args=(lhs,))

@wrap_const
def log2(lhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.LOG2, args=(lhs,))

@wrap_const
def mod(lhs: OpBase, rhs: OpBase):
    dtype = lhs.dtype
    return OpTree(dtype, OpType.MOD, args=(lhs, rhs))

@wrap_const
def clamp(arg: OpBase, lower: OpBase, upper: OpBase):
    dtype = arg.dtype
    return OpTree(dtype, OpType.CLAMP, args=(arg, lower, upper))

@wrap_const
def sign(arg: OpBase):
    return OpTree(arg.dtype, OpType.SIGN, args=(arg,))

@wrap_const
def vec2(*args: OpBase):
    return OpTree(DType(BaseType.vec2), OpType.VEC2, args=args)

@wrap_const
def vec3(*args: OpBase):
    return OpTree(DType(BaseType.vec3), OpType.VEC3, args=args)

@wrap_const
def vec4(*args: OpBase):
    return OpTree(DType(BaseType.vec4), OpType.VEC4, args=args)

@wrap_const
def mat2(*args: OpBase):
    return OpTree(DType(BaseType.mat2), OpType.MAT2, args=args)

@wrap_const
def mat3(*args: OpBase):
    return OpTree(DType(BaseType.mat3), OpType.MAT3, args=args)

@wrap_const
def mat4(*args: OpBase):
    return OpTree(DType(BaseType.mat4), OpType.MAT4, args=args)

@wrap_const
def swizzle_x(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec2: dtype = BaseType.float
        case BaseType.vec3: dtype = BaseType.float
        case BaseType.vec4: dtype = BaseType.float
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.X, args=(arg,))

@wrap_const
def swizzle_y(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec2: dtype = BaseType.float
        case BaseType.vec3: dtype = BaseType.float
        case BaseType.vec4: dtype = BaseType.float
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.Y, args=(arg,))

@wrap_const
def swizzle_z(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec3: dtype = BaseType.float
        case BaseType.vec4: dtype = BaseType.float
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.Z, args=(arg,))

@wrap_const
def swizzle_w(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec4: dtype = BaseType.float
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.W, args=(arg,))

@wrap_const
def swizzle_xy(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec2: dtype = BaseType.vec2
        case BaseType.vec3: dtype = BaseType.vec2
        case BaseType.vec4: dtype = BaseType.vec2
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.XY, args=(arg,))

@wrap_const
def swizzle_yz(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec3: dtype = BaseType.vec2
        case BaseType.vec4: dtype = BaseType.vec2
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.YZ, args=(arg,))

@wrap_const
def swizzle_xz(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec3: dtype = BaseType.vec2
        case BaseType.vec4: dtype = BaseType.vec2
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.XZ, args=(arg,))

@wrap_const
def swizzle_yzx(arg: OpBase):
    assert arg.dtype.length is None
    match arg.dtype.base:
        case BaseType.vec3: dtype = BaseType.vec3
        case BaseType.vec4: dtype = BaseType.vec3
        case _: raise NotImplementedError()
    dtype = DType(dtype)
    return OpTree(dtype, OpType.YZX, args=(arg,))

@wrap_const
def ternary(cond: OpBase, arg1: OpBase, arg2: OpBase):
    assert cond.dtype.base == BaseType.bool
    assert arg1.dtype == arg2.dtype
    return OpTree(arg1.dtype, OpType.TERNARY, args=(cond, arg1, arg2))

@wrap_const
def lt(arg1: OpBase, arg2: OpBase):
    assert arg1.dtype == arg2.dtype
    return OpTree(DType(BaseType.bool), OpType.LT, args=(arg1, arg2))

@wrap_const
def gt(arg1: OpBase, arg2: OpBase):
    assert arg1.dtype == arg2.dtype
    return OpTree(DType(BaseType.bool), OpType.GT, args=(arg1, arg2))


    