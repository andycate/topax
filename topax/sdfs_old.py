import abc

import numpy as np
from numpy.typing import ArrayLike
from typing import Any
from dataclasses import dataclass, replace
from warnings import warn

import topax.ops_old as ops_old
from topax.ops_old import Op, OpType, DType, RetType


class SDF(abc.ABC):
    def add_sdfs(self, *sdfs):
        if not hasattr(self, '_sdfs'): self._sdfs = list(sdfs)
        else: self._sdfs.extend(sdfs)

    def add_input(self, name: str, value: Any, type: RetType | DType):
        if not hasattr(self, '_values'): self._values = {}
        if not hasattr(self, '_consts'): self._consts = {}
        assert name not in self._values, f"input with name {name} already exists"
        self._values[name] = value
        v = Op(OpType.CONST, tuple(), type, value=value, sdf=self, name=name)
        self._consts[name] = v
        setattr(self, name, v)

    @abc.abstractmethod
    def sdf_definition(self, p: Op) -> Op:
        """
        Define the operations of this signed distance function.

        :param p: the current point being evaluated

        :return: Op object representing series of operations
        """
        raise NotImplementedError()

    def __call__(self, p):
        # op = self.sdf_definition(p)
        # return replace(op, sdf=self)
        return self.sdf_definition(p)
    
    def __getitem__(self, key):
        return self._values[key]
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name != '_sdfs':
            if isinstance(value, SDF) and (not hasattr(self, '_sdfs') or value not in self._sdfs):
                warn(f"sdf type {self.__class__.__name__} set an sdf attribute, but sdf has not been registered with add_sdfs!")
            elif hasattr(value, '__iter__'):
                for s in value:
                    if isinstance(s, SDF) and (not hasattr(self, '_sdfs') or s not in self._sdfs):
                        warn(f"sdf type {self.__class__.__name__} set an sdf attribute, but sdf has not been registered with add_sdfs!")
        return super().__setattr__(name, value)
    
    # def _hash(self, ctx):
    #     cls_name = self.__class__.__name__
    #     if cls_name not in ctx: ctx[cls_name] = {}
    #     if self not in ctx[cls_name]:
    #         ctx[cls_name][self] = len(ctx[cls_name].keys())
    #     return f"{cls_name}{ctx[cls_name][self]}({','.join([sdf._hash(ctx) for sdf in self._sdfs]) if hasattr(self, '_sdfs') else ''};{','.join([repr(const) for _, const in self._consts.items()]) if hasattr(self, '_consts') else ''})"
    
    # def hash(self):
    #     return self._hash({})
    



class empty(SDF):
    def sdf_definition(self, p):
        return Op(OpType.CONST, tuple(), DType.float, value='uintBitsToFloat(0x7F800000u)')

class sphere(SDF):
    def __init__(
        self,
        radius: float, 
        center: ArrayLike | None=None, 
        x: float | None=None,
        y: float | None=None,
        z: float | None=None
    ):
        if center is not None:
            self.add_input('radius', radius, DType.float)
            self.add_input('center', center, DType.vec3)
        else:
            center = (x, y, z)
            if all(e is None for e in center):
                self.add_input('radius', radius, DType.float)
            else:
                center = tuple([0.0 if e is None else e for e in center])
                self.add_input('radius', radius, DType.float)
                self.add_input('center', center, DType.vec3)

    def sdf_definition(self, p):
        if hasattr(self, 'center'):
            return ops_old.length(p - self.center) - self.radius
        else:
            return ops_old.length(p) - self.radius
        
class box(SDF):
    def __init__(
        self,
        size: ArrayLike,
    ):
        if isinstance(size, float) or len(size) == 1:
            self.add_input('size', size, DType.float)
        else:
            self.add_input('size', size, DType.vec3)

    def sdf_definition(self, p):
        q = ops_old.abs(p) - self.size
        return ops_old.length(ops_old.max(q, 0.0)) + ops_old.min(ops_old.max(q.x, ops_old.max(q.y, q.z)), 0.0)
    
class cylinder(SDF):
    def __init__(
        self,
        r: float,
        h: float,
    ):
        self.add_input('r', r, DType.float)
        self.add_input('h', h, DType.float)

    def sdf_definition(self, p):
        d = ops_old.abs(ops_old.vec2(ops_old.length(p.xz),p.y)) - ops_old.vec2(self.r,self.h)
        return ops_old.min(ops_old.max(d.x,d.y),0.0) + ops_old.length(ops_old.max(d,0.0))
    
class gyroid(SDF):
    def __init__(
        self, 
        scale: float=2.0, 
        fill: float=0.08, 
        thickness: float=0.33
    ):
        self.add_input('scale', scale, DType.float)
        self.add_input('fill', fill, DType.float)
        self.add_input('thickness', thickness, DType.float)

    def sdf_definition(self, p):
        scaled_p = p * self.scale
        gyroid = ops_old.abs(
            ops_old.dot(
                ops_old.sin(scaled_p), 
                ops_old.cos(scaled_p.yzx)
            )
        ) * self.thickness - self.fill
        return gyroid
    
class translate(SDF):
    def __init__(self, sdf: SDF, offset: ArrayLike):
        self.add_sdfs(sdf)
        self.add_input('offset', offset, DType.vec3)
        self.sdf = sdf

    def sdf_definition(self, p):
        return self.sdf(p - self.offset)
    
class rotate(SDF):
    def __init__(self, sdf: SDF, axis: str, angle: float):
        self.add_sdfs(sdf)
        self.axis = axis
        self.add_input('angle', np.deg2rad(angle), DType.float)
        self.sdf = sdf

    def sdf_definition(self, p):
        s, c = ops_old.sin(self.angle), ops_old.cos(self.angle)
        match self.axis:
            case 'x':
                rot = [
                    [1., 0., 0.], 
                    [0., c, -s], 
                    [0., s, c]
                ]
            case 'y':
                rot = [
                    [c, 0., s], 
                    [0., 1., 0.], 
                    [-s, 0., c]
                ]
            case 'z':
                rot = [
                    [c, -s, 0.], 
                    [s, c, 0.], 
                    [0., 0., 1.]
                ]
            case _:
                raise ValueError(f"Axis must be 'x' 'y' or 'z', not {self.axis}")
        p = ops_old.mat3(rot) * p
        return self.sdf(p)
    
class union(SDF):
    def __init__(self, *sdfs: SDF):
        self.add_sdfs(*sdfs)
        self.sdfs = sdfs

    def sdf_definition(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        return ops_old.min(*[sdf(p) for sdf in self.sdfs])
    
class intersect(SDF):
    def __init__(self, *sdfs: SDF):
        self.add_sdfs(*sdfs)
        self.sdfs = sdfs

    def sdf_definition(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        return ops_old.max(*[sdf(p) for sdf in self.sdfs])
    
class subtract(SDF):
    def __init__(self, sdf: SDF, tool: SDF):
        self.add_sdfs(sdf, tool)
        self.sdf = sdf
        self.tool = tool

    def sdf_definition(self, p):
        return ops_old.max(self.sdf(p), -self.tool(p))
    
class scale(SDF):
    def __init__(self, sdf: SDF, amount: float):
        self.add_sdfs(sdf)
        self.add_input('amount', amount, DType.float)
        self.sdf = sdf

    def sdf_definition(self, p):
        return self.sdf(p / self.amount) * self.amount
    
class offset(SDF):
    def __init__(self, sdf: SDF, amount: float):
        self.add_sdfs(sdf)
        self.add_input('amount', amount, DType.float)
        self.sdf = sdf

    def sdf_definition(self, p):
        return self.sdf(p) - self.amount
    
class tlp(SDF):
    """Truncated Linear Pattern"""
    def __init__(self, sdf: SDF, spacing, nrep, sym=True):
        self.add_sdfs(sdf)
        self.add_input('spacing', spacing, DType.vec3)
        self.add_input('nrep', nrep, DType.vec3)
        self.sdf = sdf
    
    def sdf_definition(self, p: Op) -> Op:
        q = p - self.spacing * ops_old.clamp(ops_old.round(p / self.spacing), -self.nrep, self.nrep)
        return self.sdf(q)
    
class cp(SDF):
    """Simple Circular Pattern"""
    def __init__(self, sdf: SDF, r: float, nrep: float):
        self.add_sdfs(sdf)
        self.add_input('r', r, DType.float)
        self.add_input('nrep', nrep, DType.float)
        self.sdf = sdf
    
    def sdf_definition(self, p: Op) -> Op:
        theta = ops_old.atan(p.y, p.x) # -np.pi to np.pi
        r = ops_old.length(p.xy)
        z = p.z
        spacing = np.pi / self.nrep
        theta_prime = theta - spacing * ops_old.clamp(ops_old.round(theta / spacing), -self.nrep, self.nrep)
        ty = ops_old.sin(theta_prime) * r
        tx = ops_old.cos(theta_prime) * r
        q = ops_old.vec3(tx - self.r, ty, z)
        return self.sdf(q)


