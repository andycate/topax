from enum import Enum
from typing import Any
from dataclasses import dataclass

class OpTypes(Enum):
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

@dataclass(frozen=True)
class Op:
    opcode: OpTypes
    lhs: Any
    rhs: Any = None
    rettype: str = ""

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
        if self.rettype == "":
            match self.opcode:
                case OpTypes.ADD: self._set_rettype()
                case OpTypes.SUB: self._set_rettype()
                case OpTypes.MUL: self._set_rettype()
                case OpTypes.DIV: self._set_rettype()
                case OpTypes.LEN: self._set_rettype('float')
                case OpTypes.NORM: self._set_rettype()
                case OpTypes.SQRT: self._set_rettype()
                case OpTypes.SIN: self._set_rettype()
                case OpTypes.COS: self._set_rettype()
                case OpTypes.MIN: self._set_rettype()
                case OpTypes.MAX: self._set_rettype()
                case OpTypes.X: self._set_rettype('float')
                case OpTypes.Y: self._set_rettype('float')
                case OpTypes.Z: self._set_rettype('float')
                case OpTypes.XY: self._set_rettype('vec2')
                case OpTypes.XZ: self._set_rettype('vec2')
                case OpTypes.YZ: self._set_rettype('vec2')
                case OpTypes.NEG: self._set_rettype(self.lhs.rettype)
                case _: raise NotImplementedError(f"rettype for opcode {self.opcode} not supported")

    def __add__(self, rhs): return Op(OpTypes.ADD, self, rhs)
    def __radd__(self, lhs): return Op(OpTypes.ADD, lhs, self)
    
    def __sub__(self, rhs): return Op(OpTypes.SUB, self, rhs)
    def __rsub__(self, lhs): return Op(OpTypes.SUB, lhs, self)

    def __pos__(self): return self
    def __neg__(self): return Op(OpTypes.NEG, self)
    
    def __mul__(self, rhs): return Op(OpTypes.MUL, self, rhs)
    def __rmul__(self, lhs): return Op(OpTypes.MUL, lhs, self)

    def __truediv__(self, rhs): return Op(OpTypes.DIV, self, rhs)
    def __rtruediv__(self, lhs): return Op(OpTypes.DIV, lhs, self)
    
    def __repr__(self):
        if self.rhs is not None:
            return f"{self.opcode}({self.lhs},{self.rhs})->{self.rettype}"
        else:
            return f"{self.opcode}({self.lhs})->{self.rettype}"
        
@dataclass(frozen=True)
class Const:
    sdf: Any
    param: str
    rettype: str

    def resolve_value(self):
        assert self.sdf is not None
        return self.sdf[self.param]

    def __add__(self, rhs): return Op(OpTypes.ADD, self, rhs)
    def __radd__(self, lhs): return Op(OpTypes.ADD, lhs, self)
    
    def __sub__(self, rhs): return Op(OpTypes.SUB, self, rhs)
    def __rsub__(self, lhs): return Op(OpTypes.SUB, lhs, self)
    
    def __pos__(self): return self
    def __neg__(self): return Op(OpTypes.NEG, self)

    def __mul__(self, rhs): return Op(OpTypes.MUL, self, rhs)
    def __rmul__(self, lhs): return Op(OpTypes.MUL, lhs, self)

    def __truediv__(self, rhs): return Op(OpTypes.DIV, self, rhs)
    def __rtruediv__(self, lhs): return Op(OpTypes.DIV, lhs, self)

    def __repr__(self): return f"Const({type(self.sdf).__name__};{self.param};{self.rettype})"

    def __eq__(self, other): return self.sdf == other.sdf and self.param == other.param
