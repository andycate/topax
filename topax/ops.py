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
    CROSS = 8
    SQUARE = 9
    SQRT = 10
    POW = 11
    SIN = 12
    COS = 13
    TAN = 14
    ASIN = 15
    ACOS = 16
    ATAN = 17
    MIN = 18
    MAX = 19
    ABS = 20
    SUBIDX = 21
    EXP = 22
    EXP2 = 23
    LOG = 24
    LOG2 = 25
    MOD = 26
    CLAMP = 27
    ROUND = 28
    FLOOR = 29
    CEIL = 30
    SIGN = 31
    VEC2 = 32
    VEC3 = 33
    VEC4 = 34
    MAT2 = 35
    MAT3 = 36
    MAT4 = 37
    X = 38
    Y = 39
    Z = 40
    W = 41
    XY = 42
    XZ = 43
    YZ = 44
    YZX = 45
    ZXY = 46
    TERNARY = 47
    LT = 48
    GT = 49

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

    def _grad_component(self, p, component):
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

    def _grad_component(self, p, component):
        # Constant has zero derivative for all components
        return const(DType(BaseType.float), 0.0)

@dataclass(frozen=True)
class param(OpBase):
    """class representing a tunable param (a uniform in a glsl shader)"""
    name: str
    value: Any = field(repr=False, hash=False)
    implicit: bool = False
    resolution: float = field(repr=False, hash=False, default=-1.)
    vmin: float = field(repr=False, hash=False, default=0.)
    vmax: float = field(repr=False, hash=False, default=1.)

    def __post_init__(self):
        if self.dtype is None: object.__setattr__(self, 'dtype', types.resolve_dtype(self.value))

    def _update_name(self, name: str):
        object.__setattr__(self, 'name', name)

    def grad(self, p):
        assert self.dtype.length is None
        assert self.dtype.base in {BaseType.float, BaseType.vec2, BaseType.vec3, BaseType.vec4}
        if self == p: return const(self.dtype, 1.0)
        else: return const(self.dtype, 0.0)

    def _grad_component(self, p, component):
        # For a vector parameter p, derivative w.r.t. component i is:
        # 1 if this is p and we're extracting component i, else 0
        if self != p:
            return const(DType(BaseType.float), 0.0)

        # This is the parameter we're differentiating with respect to
        if component is None:
            # Scalar case
            return const(DType(BaseType.float), 1.0)

        # Vector case - this should not happen directly on a param
        # The param should be accessed via swizzle operations (p.x, p.y, p.z)
        # which will handle the component extraction
        return const(DType(BaseType.float), 0.0)

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
        """
        Compute the gradient of this OpTree with respect to parameter p.

        For a scalar function f and vector parameter p (vec2 or vec3),
        returns the gradient vector (∂f/∂p.x, ∂f/∂p.y, [∂f/∂p.z]).

        This builds separate trees for each partial derivative component,
        then combines them into a vector at the end.
        """
        # Determine dimensionality from parameter type
        if p.dtype.base == BaseType.vec3:
            return vec3(
                self._grad_component(p, 0),  # ∂f/∂x
                self._grad_component(p, 1),  # ∂f/∂y
                self._grad_component(p, 2)   # ∂f/∂z
            )
        elif p.dtype.base == BaseType.vec2:
            return vec2(
                self._grad_component(p, 0),  # ∂f/∂x
                self._grad_component(p, 1)   # ∂f/∂y
            )
        else:
            # Scalar parameter - just return scalar derivative
            return self._grad_component(p, None)

    def _grad_component(self, p, component):
        """
        Compute partial derivative with respect to one component of p.

        Args:
            p: The parameter we're differentiating with respect to
            component: 0 for x, 1 for y, 2 for z, None for scalar

        Returns:
            A scalar OpTree representing the partial derivative
        """
        match self.optype:
            # Unary operations
            case OpType.NEG:
                # d(-f)/dp_i = -df/dp_i
                a = self.args[0]
                return -a._grad_component(p, component)

            case OpType.ABS:
                # d|f|/dp_i = sign(f) * df/dp_i
                a = self.args[0]
                return sign(a) * a._grad_component(p, component)

            # Binary arithmetic operations
            case OpType.ADD:
                # d(f + g)/dp_i = df/dp_i + dg/dp_i
                a, b = self.args[0], self.args[1]
                return a._grad_component(p, component) + b._grad_component(p, component)

            case OpType.SUB:
                # d(f - g)/dp_i = df/dp_i - dg/dp_i
                a, b = self.args[0], self.args[1]
                return a._grad_component(p, component) - b._grad_component(p, component)

            case OpType.MUL:
                # d(f * g)/dp_i = f * dg/dp_i + g * df/dp_i (product rule)
                a, b = self.args[0], self.args[1]
                return a * b._grad_component(p, component) + b * a._grad_component(p, component)

            case OpType.DIV:
                # d(f / g)/dp_i = (g * df/dp_i - f * dg/dp_i) / g^2 (quotient rule)
                a, b = self.args[0], self.args[1]
                return (b * a._grad_component(p, component) - a * b._grad_component(p, component)) / (b * b)

            # Min/Max operations (non-smooth, use subgradient)
            # Need to handle mixed types (e.g., vec2 and float)
            case OpType.MIN:
                # d(min(f, g))/dp_i = df/dp_i if f < g else dg/dp_i
                a, b = self.args[0], self.args[1]
                a_grad = a._grad_component(p, component)
                b_grad = b._grad_component(p, component)

                # Handle type broadcasting for comparison
                if a.dtype.base == BaseType.float and b.dtype.base == BaseType.float:
                    return ternary(a < b, a_grad, b_grad)
                elif a.dtype.base == BaseType.float:
                    # a is scalar, b is vector - result is vector, we need the right component
                    if b.dtype.base == BaseType.vec2:
                        b_comp = [b.x, b.y][component] if component is not None else b
                    elif b.dtype.base == BaseType.vec3:
                        b_comp = [b.x, b.y, b.z][component] if component is not None else b
                    else:
                        raise NotImplementedError(f"MIN gradient for {a.dtype} and {b.dtype}")
                    return ternary(a < b_comp, a_grad, b_grad)
                elif b.dtype.base == BaseType.float:
                    # a is vector, b is scalar
                    if a.dtype.base == BaseType.vec2:
                        a_comp = [a.x, a.y][component] if component is not None else a
                    elif a.dtype.base == BaseType.vec3:
                        a_comp = [a.x, a.y, a.z][component] if component is not None else a
                    else:
                        raise NotImplementedError(f"MIN gradient for {a.dtype} and {b.dtype}")
                    return ternary(a_comp < b, a_grad, b_grad)
                else:
                    # Both are vectors - compare component-wise
                    if a.dtype.base == BaseType.vec2:
                        a_comp = [a.x, a.y][component]
                        b_comp = [b.x, b.y][component]
                    elif a.dtype.base == BaseType.vec3:
                        a_comp = [a.x, a.y, a.z][component]
                        b_comp = [b.x, b.y, b.z][component]
                    else:
                        raise NotImplementedError(f"MIN gradient for {a.dtype} and {b.dtype}")
                    return ternary(a_comp < b_comp, a_grad, b_grad)

            case OpType.MAX:
                # d(max(f, g))/dp_i = df/dp_i if f > g else dg/dp_i
                a, b = self.args[0], self.args[1]
                a_grad = a._grad_component(p, component)
                b_grad = b._grad_component(p, component)

                # Handle type broadcasting for comparison
                if a.dtype.base == BaseType.float and b.dtype.base == BaseType.float:
                    return ternary(a > b, a_grad, b_grad)
                elif a.dtype.base == BaseType.float:
                    # a is scalar, b is vector
                    if b.dtype.base == BaseType.vec2:
                        b_comp = [b.x, b.y][component] if component is not None else b
                    elif b.dtype.base == BaseType.vec3:
                        b_comp = [b.x, b.y, b.z][component] if component is not None else b
                    else:
                        raise NotImplementedError(f"MAX gradient for {a.dtype} and {b.dtype}")
                    return ternary(a > b_comp, a_grad, b_grad)
                elif b.dtype.base == BaseType.float:
                    # a is vector, b is scalar
                    if a.dtype.base == BaseType.vec2:
                        a_comp = [a.x, a.y][component] if component is not None else a
                    elif a.dtype.base == BaseType.vec3:
                        a_comp = [a.x, a.y, a.z][component] if component is not None else a
                    else:
                        raise NotImplementedError(f"MAX gradient for {a.dtype} and {b.dtype}")
                    return ternary(a_comp > b, a_grad, b_grad)
                else:
                    # Both are vectors - compare component-wise
                    if a.dtype.base == BaseType.vec2:
                        a_comp = [a.x, a.y][component]
                        b_comp = [b.x, b.y][component]
                    elif a.dtype.base == BaseType.vec3:
                        a_comp = [a.x, a.y, a.z][component]
                        b_comp = [b.x, b.y, b.z][component]
                    else:
                        raise NotImplementedError(f"MAX gradient for {a.dtype} and {b.dtype}")
                    return ternary(a_comp > b_comp, a_grad, b_grad)

            case OpType.LEN:
                # d|v|/dp_i = (v · ∂v/∂p_i) / |v|
                # For vector v, this is: (v.x * ∂v.x/∂p_i + v.y * ∂v.y/∂p_i + ...) / |v|
                v = self.args[0]
                len_v = self  # length(v)

                if v.dtype.base == BaseType.vec2:
                    # Need gradients of v.x and v.y with respect to p_i
                    grad_vx = v.x._grad_component(p, component)
                    grad_vy = v.y._grad_component(p, component)
                    return (v.x * grad_vx + v.y * grad_vy) / len_v
                elif v.dtype.base == BaseType.vec3:
                    grad_vx = v.x._grad_component(p, component)
                    grad_vy = v.y._grad_component(p, component)
                    grad_vz = v.z._grad_component(p, component)
                    return (v.x * grad_vx + v.y * grad_vy + v.z * grad_vz) / len_v
                elif v.dtype.base == BaseType.vec4:
                    grad_vx = v.x._grad_component(p, component)
                    grad_vy = v.y._grad_component(p, component)
                    grad_vz = v.z._grad_component(p, component)
                    grad_vw = v.w._grad_component(p, component)
                    return (v.x * grad_vx + v.y * grad_vy + v.z * grad_vz + v.w * grad_vw) / len_v
                else:
                    raise NotImplementedError(f"LEN gradient for type {v.dtype} not supported")

            # Swizzle operations - these extract a scalar from a vector
            # The key insight: if the vector is p itself, then p.x has derivative 1 w.r.t. p.x, 0 w.r.t. p.y
            case OpType.X:
                v = self.args[0]
                if isinstance(v, param) and v == p:
                    # ∂(p.x)/∂(p.x) = 1, ∂(p.x)/∂(p.y) = 0, ∂(p.x)/∂(p.z) = 0
                    return const(DType(BaseType.float), 1.0 if component == 0 else 0.0)
                elif isinstance(v, OpTree) and v.optype == OpType.VEC2:
                    # vec2(a, b).x = a, so gradient is gradient of a
                    return v.args[0]._grad_component(p, component)
                elif isinstance(v, OpTree) and v.optype == OpType.VEC3:
                    # vec3(a, b, c).x = a
                    return v.args[0]._grad_component(p, component)
                elif isinstance(v, OpTree):
                    # For operations on vectors (ADD, SUB, MUL, etc.), distribute the swizzle
                    # (a op b).x = a.x op b.x
                    return self._distribute_swizzle_grad(v, 0, p, component)
                else:
                    return const(DType(BaseType.float), 0.0)

            case OpType.Y:
                v = self.args[0]
                if isinstance(v, param) and v == p:
                    return const(DType(BaseType.float), 1.0 if component == 1 else 0.0)
                elif isinstance(v, OpTree) and v.optype == OpType.VEC2:
                    # vec2(a, b).y = b
                    return v.args[1]._grad_component(p, component)
                elif isinstance(v, OpTree) and v.optype == OpType.VEC3:
                    # vec3(a, b, c).y = b
                    return v.args[1]._grad_component(p, component)
                elif isinstance(v, OpTree):
                    return self._distribute_swizzle_grad(v, 1, p, component)
                else:
                    return const(DType(BaseType.float), 0.0)

            case OpType.Z:
                v = self.args[0]
                if isinstance(v, param) and v == p:
                    return const(DType(BaseType.float), 1.0 if component == 2 else 0.0)
                elif isinstance(v, OpTree) and v.optype == OpType.VEC3:
                    # vec3(a, b, c).z = c
                    return v.args[2]._grad_component(p, component)
                elif isinstance(v, OpTree):
                    return self._distribute_swizzle_grad(v, 2, p, component)
                else:
                    return const(DType(BaseType.float), 0.0)

            # Vector construction operations
            # These shouldn't be called directly in _grad_component for scalar results,
            # but we need them for when vectors are used as intermediates
            case OpType.VEC2:
                # This case occurs when we need the gradient of a vec2 as a whole
                # Since _grad_component returns a scalar, this shouldn't happen
                # for well-formed scalar SDFs, but we handle it for completeness
                raise NotImplementedError("VEC2 gradient should be accessed via swizzle operations")

            case OpType.VEC3:
                raise NotImplementedError("VEC3 gradient should be accessed via swizzle operations")

            case _:
                pass

        raise NotImplementedError(f"Grad for operation {self.optype} not supported yet")

    def _distribute_swizzle_grad(self, v, swizzle_idx, p, component):
        """
        Distribute a swizzle through a vector operation and compute gradient.

        For operations like (a - b).x, this computes the gradient of (a.x - b.x).

        Args:
            v: The vector OpTree we're swizzling
            swizzle_idx: 0 for x, 1 for y, 2 for z
            p: Parameter we're differentiating with respect to
            component: Which component of p (0, 1, or 2)
        """
        def get_swizzled(arg, idx):
            """Get the appropriate swizzle of an argument"""
            if isinstance(arg, param):
                return [arg.x, arg.y, arg.z][idx] if arg.dtype.base == BaseType.vec3 else [arg.x, arg.y][idx]
            elif isinstance(arg, const):
                return arg  # Scalar constant broadcasts
            elif isinstance(arg, OpTree):
                if arg.dtype.base == BaseType.float:
                    return arg  # Scalar broadcasts
                else:
                    return [arg.x, arg.y, arg.z][idx] if arg.dtype.base == BaseType.vec3 else [arg.x, arg.y][idx]
            else:
                return arg

        # Handle different vector operations by distributing the swizzle
        match v.optype:
            case OpType.ADD:
                a_swiz = get_swizzled(v.args[0], swizzle_idx)
                b_swiz = get_swizzled(v.args[1], swizzle_idx)
                return a_swiz._grad_component(p, component) + b_swiz._grad_component(p, component)

            case OpType.SUB:
                a_swiz = get_swizzled(v.args[0], swizzle_idx)
                b_swiz = get_swizzled(v.args[1], swizzle_idx)
                return a_swiz._grad_component(p, component) - b_swiz._grad_component(p, component)

            case OpType.MUL:
                a, b = v.args[0], v.args[1]
                a_swiz = get_swizzled(a, swizzle_idx)
                b_swiz = get_swizzled(b, swizzle_idx)
                # Product rule: d(a*b)/dp = a*db/dp + b*da/dp
                return a_swiz * b_swiz._grad_component(p, component) + b_swiz * a_swiz._grad_component(p, component)

            case OpType.DIV:
                a, b = v.args[0], v.args[1]
                a_swiz = get_swizzled(a, swizzle_idx)
                b_swiz = get_swizzled(b, swizzle_idx)
                # Quotient rule
                return (b_swiz * a_swiz._grad_component(p, component) - a_swiz * b_swiz._grad_component(p, component)) / (b_swiz * b_swiz)

            case OpType.NEG:
                a_swiz = get_swizzled(v.args[0], swizzle_idx)
                return -a_swiz._grad_component(p, component)

            case OpType.ABS:
                a_swiz = get_swizzled(v.args[0], swizzle_idx)
                return sign(a_swiz) * a_swiz._grad_component(p, component)

            case OpType.MIN:
                a, b = v.args[0], v.args[1]
                a_swiz = get_swizzled(a, swizzle_idx)
                b_swiz = get_swizzled(b, swizzle_idx)
                return ternary(a_swiz < b_swiz, a_swiz._grad_component(p, component), b_swiz._grad_component(p, component))

            case OpType.MAX:
                a, b = v.args[0], v.args[1]
                a_swiz = get_swizzled(a, swizzle_idx)
                b_swiz = get_swizzled(b, swizzle_idx)
                return ternary(a_swiz > b_swiz, a_swiz._grad_component(p, component), b_swiz._grad_component(p, component))

            case _:
                raise NotImplementedError(f"Cannot distribute swizzle through operation {v.optype}")


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
    if lhs.dtype.base in {BaseType.mat2, BaseType.mat3, BaseType.mat4} and rhs.dtype.base in {BaseType.vec2, BaseType.vec3, BaseType.vec4}:
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
def cross(lhs: OpBase, rhs: OpBase):
    dtype = DType(BaseType.vec3)
    return OpTree(dtype, OpType.CROSS, args=(lhs, rhs))

@wrap_const
def dot(lhs: OpBase, rhs: OpBase):
    dtype = DType(BaseType.float)
    return OpTree(dtype, OpType.DOT, args=(lhs, rhs))

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
def round(arg: OpBase):
    dtype = arg.dtype
    return OpTree(dtype, OpType.ROUND, args=(arg,))

@wrap_const
def floor(arg: OpBase):
    dtype = arg.dtype
    return OpTree(dtype, OpType.FLOOR, args=(arg,))

@wrap_const
def ceil(arg: OpBase):
    dtype = arg.dtype
    return OpTree(dtype, OpType.CEIL, args=(arg,))

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


    