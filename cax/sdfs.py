import abc
from typing import Any
from enum import Enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
import jax.numpy as jnp
import jinja2

from cax.ops import Op, OpTypes, Const

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
    def sdf_definition(self, p) -> Op | Const:
        raise NotImplementedError()

    def __call__(self, p):
        return self.sdf_definition(p)
    
    def __getitem__(self, key):
        return self._values[key]

class empty(SDF):
    def __init__(self):
        super().__init__()

    def sdf_definition(self, p):
        return Const(None, 'uintBitsToFloat(0x7F800000u)', 'float')

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
