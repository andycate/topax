import warnings
import abc
import inspect
from typing import Any

import cax.ops as ops

class SDF(abc.ABC):
    # def __init__(self, *args, **kwargs):
    #     # here we want to figure out what type of args are being passed to compute the SDF
    #     # if any of the arguments are SDFs themselves, we need to mark this SDF as a modifier? (NOT IDEALLY)
    #     # args should be saved so that a value graph can be generated, maybe also for hashing
    #     pass
    
    def __add__(self, rhs):
        return ops.add(self, rhs)
    
    def __radd__(self, lhs):
        return ops.add(lhs, self)
    
    def __sub__(self, rhs):
        return ops.subtract(self, rhs)
    
    def __rsub__(self, lhs):
        return ops.subtract(lhs, self)

class empty(SDF):
    def sdf_definition(self, p):
        return ops.inf

class sphere(SDF):
    def __init__(self, radius: float, center: 'vec3'):
        super().__init__(radius=radius, center=center)

    def sdf_definition(self, p, center, radius):
        return ops.length(p - center) - radius
    
class translate(SDF):
    def __init__(self, sdf: SDF, offset: 'vec3'):
        super().__init__(sdf=sdf, offset=offset)

    def sdf_definition(self, p, sdf, offset):
        return sdf(p - offset)
    
class union(SDF):
    def __init__(self, *sdfs: SDF):
        super().__init__(*sdfs)

    def sdf_definition(self, p, *sdfs):
        return ops.min(*[sdf(p) for sdf in sdfs])
