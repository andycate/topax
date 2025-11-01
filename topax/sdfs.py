from functools import wraps
import inspect
from abc import abstractmethod
from ordered_set import OrderedSet
from numpy.typing import ArrayLike

import topax.ops as ops


class SDF:
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

        self._initialized = True

    def add_sdf(self, s):
        self._initialized = False
        if not hasattr(self, '_sdfs'): self._sdfs = OrderedSet()
        if s not in self._sdfs: self._sdfs.add(s)
        return s
    
    def add_param(self, p):
        self._initialized = False
        if not hasattr(self, '_explicit_params'): self._explicit_params = OrderedSet()
        if not hasattr(self, '_new_implicit_params'): self._new_implicit_params = []
        if isinstance(p, ops.const) or isinstance(p, ops.OpTree): raise TypeError(f'new param value must not be const or OpTree')
        if not isinstance(p, ops.param):
            p = ops.param(None, None, p, implicit=True)
            self._new_implicit_params.append(p)
        else:
            assert not p.implicit, "Implicit params shouldn't be passed explicitly"
            self._explicit_params.add(p)
        return p

    @abstractmethod
    def opdef(self, p: ops.OpBase) -> ops.OpBase:
        raise NotImplementedError()
    
    def __call__(self, p: ops.OpBase) -> ops.OpBase:
        if hasattr(self, '_initialized') and not self._initialized: raise ValueError('super().__init__() function must be called!')
        return self.opdef(p)
    
    def t(self, offset: ops.param=None, x: ops.param=None, y: ops.param=None, z: ops.param=None):
        """Alias for translate operation"""
        return translate(self, offset, x, y, z)
    
    # def r(self, axis: str | ops.param, angle: ops.param):
    #     """Alias for rotate operation"""
    #     return rotate(self, axis, angle)

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
            offset = ops.vec3(
                self.x if hasattr(self, 'x') else 0.0, 
                self.y if hasattr(self, 'y') else 0.0,
                self.z if hasattr(self, 'z') else 0.0,
            )
            return self.sdf(p - offset)
        
# class rotate(SDF):
#     def __init__(self, sdf: SDF, axis: str, angle: float):
#         self.add_sdfs(sdf)
#         self.axis = axis
#         self.add_input('angle', np.deg2rad(angle), DType.float)
#         self.sdf = sdf

#     def sdf_definition(self, p):
#         s, c = ops.sin(self.angle), ops.cos(self.angle)
#         match self.axis:
#             case 'x':
#                 rot = [
#                     [1., 0., 0.], 
#                     [0., c, -s], 
#                     [0., s, c]
#                 ]
#             case 'y':
#                 rot = [
#                     [c, 0., s], 
#                     [0., 1., 0.], 
#                     [-s, 0., c]
#                 ]
#             case 'z':
#                 rot = [
#                     [c, -s, 0.], 
#                     [s, c, 0.], 
#                     [0., 0., 1.]
#                 ]
#             case _:
#                 raise ValueError(f"Axis must be 'x' 'y' or 'z', not {self.axis}")
#         p = ops.mat3(rot) * p
#         return self.sdf(p)

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
    
# class intersect(SDF):
#     def __init__(self, *sdfs: SDF):
#         self.add_sdfs(*sdfs)
#         self.sdfs = sdfs

#     def sdf_definition(self, p):
#         if len(self.sdfs) == 1:
#             return self.sdfs[0](p)
#         return ops.max(*[sdf(p) for sdf in self.sdfs])
    
# class subtract(SDF):
#     def __init__(self, sdf: SDF, tool: SDF):
#         self.add_sdfs(sdf, tool)
#         self.sdf = sdf
#         self.tool = tool

#     def sdf_definition(self, p):
#         return ops.max(self.sdf(p), -self.tool(p))
    
# class scale(SDF):
#     def __init__(self, sdf: SDF, amount: float):
#         self.add_sdfs(sdf)
#         self.add_input('amount', amount, DType.float)
#         self.sdf = sdf

#     def sdf_definition(self, p):
#         return self.sdf(p / self.amount) * self.amount
    
# class offset(SDF):
#     def __init__(self, sdf: SDF, amount: float):
#         self.add_sdfs(sdf)
#         self.add_input('amount', amount, DType.float)
#         self.sdf = sdf

#     def sdf_definition(self, p):
#         return self.sdf(p) - self.amount
    
# class tlp(SDF):
#     """Truncated Linear Pattern"""
#     def __init__(self, sdf: SDF, spacing, nrep, sym=True):
#         self.add_sdfs(sdf)
#         self.add_input('spacing', spacing, DType.vec3)
#         self.add_input('nrep', nrep, DType.vec3)
#         self.sdf = sdf
    
#     def sdf_definition(self, p: Op) -> Op:
#         q = p - self.spacing * ops.clamp(ops.round(p / self.spacing), -self.nrep, self.nrep)
#         return self.sdf(q)
    
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
