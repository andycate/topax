from enum import Enum
from typing import Any
from dataclasses import dataclass

from topax._utils import resolve_type

# TODO: make this enum numbering better
class OpType(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    LEN = 5
    NORM = 6
    SQRT = 7
    SIN = 8
    COS = 9
    MIN = 10
    MAX = 11
    X = 12
    Y = 13
    Z = 14
    XY = 15
    XZ = 16
    YZ = 17
    NEG = 18
    ABS = 19

@dataclass(frozen=True)
class Op:
    opcode: OpType
    lhs: Any
    rhs: Any = None
    rettype: str = ""

    # TODO: improve this logic to be more robust
    def _set_rettype(self, rettype=None):
        if rettype is not None:
            object.__setattr__(self, 'rettype', rettype)
        else:
            lhs_rettype = self.lhs.rettype
            rhs_rettype = self.rhs.rettype if self.rhs is not None else ''
            if lhs_rettype == 'vec3' or rhs_rettype == 'vec3':
                object.__setattr__(self, 'rettype', 'vec3')
            elif lhs_rettype == 'vec2' or rhs_rettype == 'vec2':
                object.__setattr__(self, 'rettype', 'vec2')
            else:
                object.__setattr__(self, 'rettype', 'float')

    def __post_init__(self):
        if not isinstance(self.lhs, Op) and not isinstance(self.lhs, Const):
            object.__setattr__(self, 'lhs', Const(None, self.lhs))
        if self.rhs is not None and not isinstance(self.rhs, Op) and not isinstance(self.rhs, Const):
            object.__setattr__(self, 'rhs', Const(None, self.rhs))
        if self.rettype == "":
            match self.opcode:
                case OpType.ADD: self._set_rettype()
                case OpType.SUB: self._set_rettype()
                case OpType.MUL: self._set_rettype()
                case OpType.DIV: self._set_rettype()
                case OpType.LEN: self._set_rettype('float')
                case OpType.NORM: self._set_rettype()
                case OpType.SQRT: self._set_rettype()
                case OpType.SIN: self._set_rettype()
                case OpType.COS: self._set_rettype()
                case OpType.MIN: self._set_rettype()
                case OpType.MAX: self._set_rettype()
                case OpType.X: self._set_rettype('float')
                case OpType.Y: self._set_rettype('float')
                case OpType.Z: self._set_rettype('float')
                case OpType.XY: self._set_rettype('vec2')
                case OpType.XZ: self._set_rettype('vec2')
                case OpType.YZ: self._set_rettype('vec2')
                case OpType.NEG: self._set_rettype(self.lhs.rettype)
                case OpType.ABS: self._set_rettype(self.lhs.rettype)
                case _: raise NotImplementedError(f"rettype for opcode {self.opcode} not supported")

    @property
    def x(self): return Op(OpType.X, self)
    @property
    def y(self): return Op(OpType.Y, self)
    @property
    def z(self): return Op(OpType.Z, self)
    @property
    def xy(self): return Op(OpType.XY, self)
    @property
    def xz(self): return Op(OpType.XZ, self)
    @property
    def yz(self): return Op(OpType.YZ, self)

    def __add__(self, rhs): return Op(OpType.ADD, self, rhs)
    def __radd__(self, lhs): return Op(OpType.ADD, lhs, self)
    
    def __sub__(self, rhs): return Op(OpType.SUB, self, rhs)
    def __rsub__(self, lhs): return Op(OpType.SUB, lhs, self)

    def __pos__(self): return self
    def __neg__(self): return Op(OpType.NEG, self)
    
    def __mul__(self, rhs): return Op(OpType.MUL, self, rhs)
    def __rmul__(self, lhs): return Op(OpType.MUL, lhs, self)

    def __truediv__(self, rhs): return Op(OpType.DIV, self, rhs)
    def __rtruediv__(self, lhs): return Op(OpType.DIV, lhs, self)
    
    def __repr__(self):
        if self.rhs is not None:
            return f"{self.opcode}({self.lhs},{self.rhs})->{self.rettype}"
        else:
            return f"{self.opcode}({self.lhs})->{self.rettype}"

# TODO: maybe get rid of this and just make it an OpType?    
@dataclass(frozen=True)
class Const:
    sdf: Any
    param: str | Any
    rettype: str | None = None

    def __post_init__(self):
        if self.rettype is None:
            type = resolve_type(self.param)
            object.__setattr__(self, 'rettype', type)

    def resolve_value(self):
        assert self.sdf is not None
        return self.sdf[self.param]
    
    @property
    def x(self): return Op(OpType.X, self)
    @property
    def y(self): return Op(OpType.Y, self)
    @property
    def z(self): return Op(OpType.Z, self)
    @property
    def xy(self): return Op(OpType.XY, self)
    @property
    def xz(self): return Op(OpType.XZ, self)
    @property
    def yz(self): return Op(OpType.YZ, self)

    def __add__(self, rhs): return Op(OpType.ADD, self, rhs)
    def __radd__(self, lhs): return Op(OpType.ADD, lhs, self)
    
    def __sub__(self, rhs): return Op(OpType.SUB, self, rhs)
    def __rsub__(self, lhs): return Op(OpType.SUB, lhs, self)
    
    def __pos__(self): return self
    def __neg__(self): return Op(OpType.NEG, self)

    def __mul__(self, rhs): return Op(OpType.MUL, self, rhs)
    def __rmul__(self, lhs): return Op(OpType.MUL, lhs, self)

    def __truediv__(self, rhs): return Op(OpType.DIV, self, rhs)
    def __rtruediv__(self, lhs): return Op(OpType.DIV, lhs, self)

    def __repr__(self): return f"Const({type(self.sdf).__name__};{self.param};{self.rettype})"

    def __eq__(self, other): return self.sdf == other.sdf and self.param == other.param
