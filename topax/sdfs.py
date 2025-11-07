from functools import wraps
import inspect
from abc import abstractmethod
from ordered_set import OrderedSet
import numpy as np
from numpy.typing import ArrayLike

import topax.ops as ops


class SDF:
    IS_2D = None
    def __init__(self):
        self._implicit_params = []
        if not hasattr(self, '_explicit_params'): self._explicit_params = OrderedSet()
        if hasattr(self, '_sdfs'):
            for s in self._sdfs:
                self._implicit_params.extend(s._implicit_params)
                self._explicit_params.update(s._explicit_params)

        for i, p in enumerate(self._implicit_params):
            p._update_name(f'_implicit_param_{i}')

        if hasattr(self, '_new_implicit_params'):
            for p in self._new_implicit_params:
                p._update_name(f'_implicit_param_{len(self._implicit_params)}')
                self._implicit_params.append(p)

        if not hasattr(self, '_sdfs'): self._sdfs = OrderedSet()
        self._initialized = True

    def add_sdf(self, s):
        self._initialized = False
        if not hasattr(self, '_sdfs'): self._sdfs = OrderedSet()
        if s not in self._sdfs: self._sdfs.add(s)
        return s
    
    def add_param(self, p, dtype=None):
        self._initialized = False
        if not hasattr(self, '_explicit_params'): self._explicit_params = OrderedSet()
        if not hasattr(self, '_new_implicit_params'): self._new_implicit_params = []
        if isinstance(p, ops.const) or isinstance(p, ops.OpTree): raise TypeError(f'new param value must not be const or OpTree')
        if not isinstance(p, ops.param):
            p = ops.param(dtype, None, p, implicit=True)
            self._new_implicit_params.append(p)
        else:
            assert not p.implicit, "Implicit params shouldn't be passed explicitly"
            self._explicit_params.add(p)
        return p

    @abstractmethod
    def opdef(self, p: ops.OpBase) -> ops.OpBase:
        raise NotImplementedError()
    
    @property
    def is_2d(self):
        assert self._initialized
        if self.IS_2D: return True
        else: return any([s.is_2d for s in self._sdfs])
    
    def __call__(self, p: ops.OpBase) -> ops.OpBase:
        if hasattr(self, '_initialized') and not self._initialized: raise ValueError('super().__init__() function must be called!')
        return self.opdef(p)
    
    def t(self, offset: ops.param=None, x: ops.param=None, y: ops.param=None, z: ops.param=None):
        """Alias for translate operation"""
        return translate(self, offset, x, y, z)
    
    def r(self, angle: ops.param, axis: str = 'x'):
        """Alias for translate operation"""
        return rotate(self, angle=angle, axis=axis)
    
    # def r(self, axis: str | ops.param, angle: ops.param):
    #     """Alias for rotate operation"""
    #     return rotate(self, axis, angle)

    def s(self, amount: float): return scale(self, amount)
    def o(self, amount: float): return offset(self, amount)
    def i(self, other): return intersect(self, other)
    def u(self, other): return union(self, other)
    def sub(self, other): return subtract(self, other)

class translate(SDF):
    def __init__(self, sdf: SDF, offset: ops.param=None, x: ops.param=None, y: ops.param=None, z: ops.param=None):
        self.sdf = self.add_sdf(sdf)
        if offset is not None:
            assert x is None and y is None and z is None, 'Cannot pass vector offset and individual offsets'
            self.offset = self.add_param(offset)
        else:
            if x is not None: self.x = self.add_param(x)
            if y is not None: self.y = self.add_param(y)
            if z is not None: self.z = self.add_param(z)
        super().__init__()

    def opdef(self, p: ops.OpBase):
        if hasattr(self, 'offset'):
            return self.sdf(p - self.offset)
        else:
            if self.sdf.is_2d:
                offset = ops.vec2(
                    self.x if hasattr(self, 'x') else 0.0, 
                    self.y if hasattr(self, 'y') else 0.0,
                )
            else:
                offset = ops.vec3(
                    self.x if hasattr(self, 'x') else 0.0, 
                    self.y if hasattr(self, 'y') else 0.0,
                    self.z if hasattr(self, 'z') else 0.0,
                )
            return self.sdf(p - offset)
        
class rotate(SDF):
    def __init__(self, sdf: SDF, angle: float, axis: str='x'):
        self.sdf = self.add_sdf(sdf)
        self.axis = axis
        self.angle = self.add_param(np.deg2rad(angle))
        super().__init__()

    def opdef(self, p):
        s, c = ops.sin(self.angle), ops.cos(self.angle)
        if self.is_2d:
            rot = [
                c, s, 
                -s, c
            ]
            p = ops.mat2(*rot) * p
            return self.sdf(p)
        else:
            match self.axis:
                case 'x':
                    rot = [
                        1., 0., 0., 
                        0., c, s, 
                        0., -s, c
                    ]
                case 'y':
                    rot = [
                        c, 0., -s, 
                        0., 1., 0., 
                        s, 0., c
                    ]
                case 'z':
                    rot = [
                        c, s, 0., 
                        -s, c, 0., 
                        0., 0., 1.
                    ]
                case _:
                    raise ValueError(f"Axis must be 'x' 'y' or 'z', not {self.axis}")
            p = ops.mat3(*rot) * p
            return self.sdf(p)

class union(SDF):
    def __init__(self, *sdfs: SDF):
        self.sdfs = []
        for sdf in sdfs:
            self.sdfs.append(self.add_sdf(sdf))
        super().__init__()

    def opdef(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        accum = self.sdfs[0](p)
        for i in range(1, len(self.sdfs)):
            accum = ops.min(accum, self.sdfs[i](p))
        return accum
    
class intersect(SDF):
    def __init__(self, *sdfs: SDF):
        sdfs_added = []
        for sdf in sdfs:
            sdfs_added.append(self.add_sdf(sdf))
        self.sdfs = sdfs_added
        super().__init__()

    def opdef(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        result = ops.max(self.sdfs[0](p), self.sdfs[1](p))
        for i in range(2, len(self.sdfs)):
            result = ops.max(result, self.sdfs[i](p))
        return result
    
class subtract(SDF):
    def __init__(self, sdf: SDF, tool: SDF):
        self.sdf = self.add_sdf(sdf)
        self.tool = self.add_sdf(tool)
        super().__init__()

    def opdef(self, p):
        return ops.max(self.sdf(p), -self.tool(p))
    
class scale(SDF):
    def __init__(self, sdf: SDF, amount: float):
        self.amount = self.add_param(amount)
        self.sdf = self.add_sdf(sdf)
        super().__init__()

    def opdef(self, p):
        return self.sdf(p / self.amount) * self.amount
    
class offset(SDF):
    def __init__(self, sdf: SDF, amount: float):
        self.amount = self.add_param(amount)
        self.sdf = self.add_sdf(sdf)
        super().__init__()

    def opdef(self, p):
        return self.sdf(p) - self.amount
    
class tlp(SDF):
    """Truncated Linear Pattern"""
    def __init__(self, sdf: SDF, spacing, n_repeats, axis='x'):
        self.sdf = self.add_sdf(sdf)
        self.spacing = self.add_param(spacing)
        self.n_repeats = self.add_param(n_repeats)
        self.axis = axis
        assert axis in {'x', 'y', 'z'}
        super().__init__()

    def opdef(self, p: ops.OpBase):
        if self.axis == 'x':
            q = p.x - self.spacing * ops.clamp(ops.round(p.x / self.spacing), -self.n_repeats, self.n_repeats)
            if self.is_2d: p = ops.vec2(q, p.y)
            else: p = ops.vec3(q, p.yz)
        elif self.axis == 'y':
            q = p.y - self.spacing * ops.clamp(ops.round(p.y / self.spacing), -self.n_repeats, self.n_repeats)
            if self.is_2d: p = ops.vec2(p.x, q)
            else: p = ops.vec3(p.x, q, p.z)
        elif self.axis == 'z':
            q = p.z - self.spacing * ops.clamp(ops.round(p.z / self.spacing), -self.n_repeats, self.n_repeats)
            p = ops.vec3(p.xy, q)
        return self.sdf(p)
    
# class cp(SDF):
#     """Simple Circular Pattern"""
#     def __init__(self, sdf: SDF, r: float, nrep: float):
#         self.add_sdfs(sdf)
#         self.add_input('r', r, DType.float)
#         self.add_input('nrep', nrep, DType.float)
#         self.sdf = sdf
    
#     def sdf_definition(self, p: Op) -> Op:
#         theta = ops.atan(p.y, p.x) # -np.pi to np.pi
#         r = ops.length(p.xy)
#         z = p.z
#         spacing = np.pi / self.nrep
#         theta_prime = theta - spacing * ops.clamp(ops.round(theta / spacing), -self.nrep, self.nrep)
#         ty = ops.sin(theta_prime) * r
#         tx = ops.cos(theta_prime) * r
#         q = ops.vec3(tx - self.r, ty, z)
#         return self.sdf(q)

class slice_2d(SDF):
    IS_2D = True
    def __init__(self, sdf: SDF, z: float=0.0):
        self.z_value = self.add_param(z)
        self.sdf = self.add_sdf(sdf)
        super().__init__()

    def opdef(self, p):
        return self.sdf(ops.vec3(p, self.z_value))

class sphere(SDF):
    def __init__(self, radius: ops.param | float):
        self.radius = self.add_param(radius)
        super().__init__()

    def opdef(self, p: ops.OpBase):
        return ops.length(p) - self.radius
    
class box(SDF):
    def __init__(
        self,
        x: ops.param | ArrayLike,
        y: ops.param | ArrayLike = None,
        z: ops.param | ArrayLike = None,
    ):
        assert (y is None) == (z is None), "either provide one value or all three axes"
        self.x = self.add_param(x)
        if y is not None:
            self.y = self.add_param(y)
            self.z = self.add_param(z)
        else:
            self.y = self.x
            self.z = self.x
        
        super().__init__()

    def opdef(self, p):
        q = ops.abs(p) - ops.vec3(self.x, self.y, self.z)
        return ops.length(ops.max(q, 0.0)) + ops.min(ops.max(q.x, ops.max(q.y, q.z)), 0.0)
    
class cylinder(SDF):
    def __init__(
        self,
        radius: ops.param | float,
        height: ops.param | float,
    ):
        self.radius = self.add_param(radius)
        self.height = self.add_param(height)
        super().__init__()

    def opdef(self, p):
        d = ops.abs(ops.vec2(ops.length(p.xz),p.y)) - ops.vec2(self.radius,self.height)
        return ops.min(ops.max(d.x,d.y),0.0) + ops.length(ops.max(d,0.0))
    
class gyroid(SDF):
    def __init__(
        self, 
        scale: float=2.0, 
        fill: float=0.08, 
        thickness: float=0.33
    ):
        self.scale = self.add_param(scale)
        self.fill = self.add_param(fill)
        self.thickness = self.add_param(thickness)
        super().__init__()

    def opdef(self, p):
        scaled_p = p * self.scale
        gyroid = ops.abs(
            ops.dot(
                ops.sin(scaled_p), 
                ops.cos(scaled_p.yzx)
            )
        ) * self.thickness - self.fill
        return gyroid
    
class circle(SDF):
    IS_2D = True
    def __init__(
        self,
        radius: ops.param | float,
    ):
        self.radius = self.add_param(radius)
        super().__init__()

    def opdef(self, p):
        return ops.length(p) - self.radius

class rectangle(SDF):
    IS_2D = True
    def __init__(
        self,
        x: ops.param | float,
        y: ops.param | float = None,
    ):
        self.x_length = self.add_param(x)
        if y is not None: self.y_length = self.add_param(y)
        super().__init__()

    def opdef(self, p):
        if hasattr(self, 'y_length'): q = ops.abs(p) - ops.vec2(self.x_length, self.y_length)
        else: q = ops.abs(p) - self.x_length
        return ops.length(ops.max(q, 0.0)) + ops.min(ops.max(q.x, q.y), 0.0)
