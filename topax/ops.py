from enum import Enum
from typing import Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp

class DType(Enum):
    vec2 = 1
    vec3 = 2
    vec4 = 3
    float = 4


@dataclass(frozen=True)
class RetType:
    dtype: DType
    length: int | None = None

    @staticmethod
    def resolve_type(item: Any):
        if hasattr(item, '__iter__'):
            l = len(item)
            item = item.item() if hasattr(item, 'item') else item[0]
        else:
            l = None
        
        raw_type = None
        if hasattr(item, 'dtype'):
            raw_type_name = item.dtype.name
            if raw_type_name.find('float') > -1: raw_type = DType.float
        else:
            if isinstance(item, float): raw_type = DType.float
        match l:
            case 2:
                l = None
                if raw_type == DType.float: raw_type = DType.vec2
            case 3:
                l = None
                if raw_type == DType.float: raw_type = DType.vec3
            case 4:
                l = None
                if raw_type == DType.float: raw_type = DType.vec4
        if raw_type is None:
            raise TypeError(f"type not detected for value {item}")
        return RetType(raw_type, l)
        


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
    MIN = 14
    MAX = 15
    NEG = 16
    ABS = 17
    DOT = 18
    X = 19
    Y = 20
    Z = 21
    XY = 22
    XZ = 23
    YZ = 24
    YZX = 25
    ZXY = 26

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
    
    def __repr__(self):
        return f"{self.opcode}({self.name};{repr(self.sdf) if self.sdf is not None else ''};{','.join([repr(arg) for arg in self.args])})->{self.rettype}"

def length(arg): return Op(OpType.LEN, (arg,))
def min(*args): return Op(OpType.MIN, tuple(args))
def max(*args): return Op(OpType.MAX, tuple(args))
def abs(arg): return Op(OpType.ABS, (arg,))
def dot(arg1, arg2): return Op(OpType.DOT, (arg1, arg2))
def sin(arg): return Op(OpType.SIN, (arg,))
def cos(arg): return Op(OpType.COS, (arg,))
