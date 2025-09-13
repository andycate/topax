import warnings
import abc
import inspect
from typing import Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import jax.numpy as jnp

# import cax.ops as ops

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
                case _: raise NotImplementedError(f"rettype for opcode {self.opcode} not supported")

    def __add__(self, rhs):
        return Op(OpTypes.ADD, self, rhs)
    
    def __radd__(self, lhs):
        return Op(OpTypes.ADD, lhs, self)
    
    def __sub__(self, rhs):
        return Op(OpTypes.SUB, self, rhs)
    
    def __rsub__(self, lhs):
        return Op(OpTypes.SUB, lhs, self)
    
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

    def __add__(self, rhs):
        return Op(OpTypes.ADD, self, rhs)
    
    def __radd__(self, lhs):
        return Op(OpTypes.ADD, lhs, self)
    
    def __sub__(self, rhs):
        return Op(OpTypes.SUB, self, rhs)
    
    def __rsub__(self, lhs):
        return Op(OpTypes.SUB, lhs, self)

    def __repr__(self):
        return f"Const({type(self.sdf).__name__};{self.rettype};{self.param})"

class SDF(abc.ABC):
    def __init__(self, *args, **kwargs):
        assert len(args) == 0
        self._values = {}
        for k in kwargs:
            item = kwargs[k]
            self._values[k] = item
            if isinstance(item, float):
                type = 'float'
            elif isinstance(item, list) or isinstance(item, tuple) or isinstance(item, np.ndarray) or isinstance(item, jnp.ndarray):
                match len(item):
                    case 3:
                        type = 'vec3'
                    case 2:
                        type = 'vec2'
                    case 1:
                        type = 'float'
                    case _:
                        raise ValueError("only float, vec3 and vec2 are supported currently")
            else:
                raise ValueError("only float, vec3 and vec2 are supported currently")
            v = Const(self, k, type)
            setattr(self, k, v)

    @abc.abstractmethod
    def sdf_definition(self, p) -> Op:
        raise NotImplementedError()

    def __call__(self, p):
        return self.sdf_definition(p)

class sphere(SDF):
    def __init__(self, radius: float, center: ArrayLike):
        super().__init__(radius=radius, center=center)

    def sdf_definition(self, p):
        return Op(OpTypes.LEN, p - self.center) - self.radius
    
class translate(SDF):
    def __init__(self, sdf: SDF, offset: ArrayLike):
        self.sdf = sdf
        super().__init__(offset=offset)

    def sdf_definition(self, p):
        return self.sdf(p - self.offset)
    
class union(SDF):
    def __init__(self, *sdfs: SDF):
        self.sdfs = sdfs

    def sdf_definition(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        oper = Op(OpTypes.MIN, self.sdfs[0](p), self.sdfs[1](p))
        for i in range(2, len(self.sdfs)):
            oper = Op(OpTypes.MIN, oper, self.sdfs[i](p))
        return oper
