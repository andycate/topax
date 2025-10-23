from enum import Enum
from typing import Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp

class DType(Enum):
    ivec2 = 1
    ivec3 = 2
    ivec4 = 3
    vec2 = 4
    vec3 = 5
    vec4 = 6
    mat2 = 7
    mat3 = 7
    mat4 = 8
    float = 9
    int = 10


@dataclass(frozen=True)
class RetType:
    dtype: DType
    length: int | None = None

    @staticmethod
    def resolve_type(item: Any):
        if hasattr(item, 'shape'): s = item.shape
        elif hasattr(item, '__iter__'): s = (len(item),)
        else: s = tuple()
        
        raw_type = None
        if hasattr(item, 'dtype'):
            raw_type_name = item.dtype.name
            if raw_type_name.find('float') > -1: raw_type = DType.float
            elif raw_type_name.find('int') > -1: raw_type = DType.int
        elif len(s) > 0:
            if isinstance(item[0], float): raw_type = DType.float
            elif isinstance(item[0], int): raw_type = DType.int
        else:
            if isinstance(item, float): raw_type = DType.float
            elif isinstance(item, int): raw_type = DType.int

        if raw_type == DType.float:
            if len(s) == 0: return RetType(DType.float, None)
            elif len(s) == 1 and s[0] == 2: return RetType(DType.vec2, None)
            elif len(s) == 1 and s[0] == 3: return RetType(DType.vec3, None)
            elif len(s) == 1 and s[0] == 4: return RetType(DType.vec4, None)
            elif len(s) == 2 and s[0] == 2 and s[1] == 2: return RetType(DType.mat2, None)
            elif len(s) == 2 and s[0] == 3 and s[1] == 3: return RetType(DType.mat3, None)
            elif len(s) == 2 and s[0] == 4 and s[1] == 4: return RetType(DType.mat4, None)
            elif len(s) == 1: return RetType(DType.float, s[0])
        if raw_type == DType.int:
            if len(s) == 0: return RetType(DType.int, None)
            elif len(s) == 1 and s[0] == 2: return RetType(DType.ivec2, None)
            elif len(s) == 1 and s[0] == 3: return RetType(DType.ivec3, None)
            elif len(s) == 1 and s[0] == 4: return RetType(DType.ivec4, None)
            elif len(s) == 1: return RetType(DType.int, s[0])
        
        raise TypeError(f"type not detected for value {item}")
        


# TODO: make this enum numbering better
class OpType(Enum):
    CONST = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    LEN = 5
    NORM = 6
    SQRT = 7
    SIN = 8
    COS = 9
    TAN = 10
    ASIN = 11
    ACOS = 12
    ATAN = 13
    MIN = 15
    MAX = 16
    NEG = 17
    ABS = 18
    DOT = 19
    X = 20
    Y = 21
    Z = 22
    XY = 23
    XZ = 24
    YZ = 25
    YZX = 26
    ZXY = 27
    VEC2 = 28
    SUBIDX = 29

@dataclass(frozen=True)
class Op:
    opcode: OpType
    args: Tuple[Any]
    rettype: RetType | None = None
    sdf: Any | None = None
    value: Any | None = field(default=None, hash=False, compare=False)
    name: str = ""

    # TODO: improve this logic to be more robust
    def _set_rettype(self, rettype=None):
        if rettype is not None:
            if isinstance(rettype, DType):
                rettype = RetType(rettype)
            assert isinstance(rettype, RetType)
            object.__setattr__(self, 'rettype', rettype)
        else:
            assert not any([arg.rettype.length is not None for arg in self.args]), "auto type resolving logic won't work for arrays"
            typerank = {
                None: 0,
                DType.float: 1,
                DType.vec2: 2,
                DType.vec3: 3,
                DType.vec4: 4,
            }
            for arg in self.args:
                assert arg.rettype.dtype in typerank, f"can't auto resolve type for {arg.rettype}"
                if typerank[arg.rettype.dtype] > typerank[rettype]:
                    rettype = arg.rettype.dtype
            assert rettype is not None
            object.__setattr__(self, 'rettype', RetType(rettype))

    def __post_init__(self):
        args = self.args
        if not hasattr(args, '__iter__'):
            args = [args]
        if self.opcode != OpType.CONST:
            args = [Op(OpType.CONST, (arg,), RetType.resolve_type(arg), value=arg) if not isinstance(arg, Op) else arg for arg in args]
        object.__setattr__(self, 'args', tuple(args))
        if isinstance(self.rettype, DType):
            object.__setattr__(self, 'rettype', RetType(self.rettype))
        elif self.rettype == None:
            match self.opcode:
                case OpType.ADD: self._set_rettype()
                case OpType.SUB: self._set_rettype()
                case OpType.MUL: self._set_rettype()
                case OpType.DIV: self._set_rettype()
                case OpType.LEN: self._set_rettype(DType.float)
                case OpType.NORM: self._set_rettype()
                case OpType.SQRT: self._set_rettype()
                case OpType.SIN: self._set_rettype()
                case OpType.COS: self._set_rettype()
                case OpType.TAN: self._set_rettype()
                case OpType.ASIN: self._set_rettype()
                case OpType.ACOS: self._set_rettype()
                case OpType.ATAN: self._set_rettype()
                case OpType.MIN: self._set_rettype()
                case OpType.MAX: self._set_rettype()
                case OpType.X: self._set_rettype(DType.float)
                case OpType.Y: self._set_rettype(DType.float)
                case OpType.Z: self._set_rettype(DType.float)
                case OpType.XY: self._set_rettype(DType.vec2)
                case OpType.XZ: self._set_rettype(DType.vec2)
                case OpType.YZ: self._set_rettype(DType.vec2)
                case OpType.YZX: self._set_rettype(DType.vec3)
                case OpType.ZXY: self._set_rettype(DType.vec3)
                case OpType.DOT: self._set_rettype(DType.float)
                case OpType.NEG: self._set_rettype(self.args[0].rettype)
                case OpType.ABS: self._set_rettype(self.args[0].rettype)
                case OpType.VEC2: self._set_rettype(DType.vec2)
                case _: raise NotImplementedError(f"rettype for opcode {self.opcode} not supported")

    @property
    def x(self): return Op(OpType.X, (self,))
    @property
    def y(self): return Op(OpType.Y, (self,))
    @property
    def z(self): return Op(OpType.Z, (self,))
    @property
    def xy(self): return Op(OpType.XY, (self,))
    @property
    def xz(self): return Op(OpType.XZ, (self,))
    @property
    def yz(self): return Op(OpType.YZ, (self,))
    @property
    def yzx(self): return Op(OpType.YZX, (self,))
    @property
    def zxy(self): return Op(OpType.ZXY, (self,))

    def __add__(self, rhs): return Op(OpType.ADD, (self, rhs))
    def __radd__(self, lhs): return Op(OpType.ADD, (lhs, self))
    
    def __sub__(self, rhs): return Op(OpType.SUB, (self, rhs))
    def __rsub__(self, lhs): return Op(OpType.SUB, (lhs, self))

    def __pos__(self): return self
    def __neg__(self): return Op(OpType.NEG, (self,))
    
    def __mul__(self, rhs): return Op(OpType.MUL, (self, rhs))
    def __rmul__(self, lhs): return Op(OpType.MUL, (lhs, self))

    def __truediv__(self, rhs): return Op(OpType.DIV, (self, rhs))
    def __rtruediv__(self, lhs): return Op(OpType.DIV, (lhs, self))

    def __getitem__(self, key):
        assert self.rettype.length is not None
        assert isinstance(key, int)
        assert key < self.rettype.length
        return Op(OpType.SUBIDX, (self, Op(OpType.CONST, (key,), DType.int, value=key)), rettype=self.rettype.dtype)
    
    def __repr__(self):
        return f"{self.opcode}({self.name};{','.join([repr(arg) for arg in self.args])})->{self.rettype}"

def length(arg): return Op(OpType.LEN, (arg,))
def min(*args): return Op(OpType.MIN, tuple(args))
def max(*args): return Op(OpType.MAX, tuple(args))
def abs(arg): return Op(OpType.ABS, (arg,))
def dot(arg1, arg2): return Op(OpType.DOT, (arg1, arg2))
def sin(arg): return Op(OpType.SIN, (arg,))
def cos(arg): return Op(OpType.COS, (arg,))
def vec2(*args): return Op(OpType.VEC2, args)
def atan(*args): return Op(OpType.ATAN, args)
