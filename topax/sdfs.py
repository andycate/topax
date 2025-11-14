from functools import wraps
import inspect
from abc import abstractmethod
from ordered_set import OrderedSet
import numpy as np
from numpy.typing import ArrayLike

import topax.ops as ops
from topax.types import DType, BaseType


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
        if isinstance(p, ops.const): raise TypeError(f'new param value must not be const')
        elif isinstance(p, ops.OpTree):
            for a in p.args:
                if not isinstance(a, ops.const): self.add_param(a)
        elif not isinstance(p, ops.param):
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
        if self.IS_2D is not None: return self.IS_2D
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
    
    def extrude(self, height: ops.param, axis='z', sym: bool=False):
        """Alias for extrude operation"""
        assert self.is_2d, "Can only extrude 2D shapes"
        return extrude(self, height, axis, sym)
    
    # def r(self, axis: str | ops.param, angle: ops.param):
    #     """Alias for rotate operation"""
    #     return rotate(self, axis, angle)

    def s(self, amount: float): return scale(self, amount)
    def o(self, amount: float): return offset(self, amount)
    def i(self, other): return intersect(self, other)
    def u(self, other): return union(self, other)
    def sub(self, other): return subtract(self, other)
    def tlp(self, spacing: float, nrep: float, axis='x', sym=True): return tlp(self, spacing, nrep, axis, sym)
    def cp(self, nrep: float, axis='x'): return cp(self, nrep, axis)

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
        self.angle = self.add_param(angle)
        super().__init__()

    def opdef(self, p):
        angle = self.angle * (np.pi / 180.)
        s, c = ops.sin(angle), ops.cos(angle)
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
    def __init__(self, sdf: SDF, spacing, n_repeats, axis='x', sym=True):
        self.sdf = self.add_sdf(sdf)
        self.spacing = self.add_param(spacing)
        self.n_repeats = self.add_param(n_repeats)
        self.axis = axis
        self.is_sym = sym
        assert axis in {'x', 'y', 'z'}
        super().__init__()

    def opdef(self, p: ops.OpBase):
        nrep = (self.n_repeats-1.)
        if self.axis == 'x':
            q = p.x
            if not self.is_sym: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), 0., nrep)
            else: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), -nrep/2., nrep/2.)
            if self.is_2d: p = ops.vec2(q, p.y)
            else: p = ops.vec3(q, p.yz)
        elif self.axis == 'y':
            q = p.y
            if not self.is_sym: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), 0., nrep)
            else: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), -nrep/2., nrep/2.)
            if self.is_2d: p = ops.vec2(p.x, q)
            else: p = ops.vec3(p.x, q, p.z)
        elif self.axis == 'z':
            q = p.z
            if not self.is_sym: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), 0., nrep)
            else: q = q - self.spacing * ops.clamp(ops.round(q / self.spacing), -nrep/2., nrep/2.)
            p = ops.vec3(p.xy, q)
        return self.sdf(p)
    
class cp(SDF):
    """Simple Circular Pattern"""
    def __init__(self, sdf: SDF, nrep: float, axis='x'):
        self.sdf = self.add_sdf(sdf)
        self.nrep = self.add_param(nrep, DType(BaseType.float))
        self.axis = axis
        super().__init__()
    
    def opdef(self, p):
        if self.sdf.is_2d:
            theta = ops.atan(p.y, p.x) # -np.pi to np.pi
            r = ops.length(p.xy)
            _nrep = self.nrep / 2.0
            spacing = np.pi / _nrep
            theta_prime = theta - spacing * ops.clamp(ops.round(theta / spacing), -_nrep, _nrep)
            ty = ops.sin(theta_prime) * r
            tx = ops.cos(theta_prime) * r
            return self.sdf(ops.vec2(tx, ty))
        if self.axis == 'x':
            theta = ops.atan(p.z, p.y) # -np.pi to np.pi
            r = ops.length(p.yz)
            _nrep = self.nrep / 2.0
            spacing = np.pi / _nrep
            theta_prime = theta - spacing * ops.clamp(ops.round(theta / spacing), -_nrep, _nrep)
            ty = ops.sin(theta_prime) * r
            tx = ops.cos(theta_prime) * r
            q = ops.vec3(p.x, tx, ty)
        elif self.axis == 'y':
            theta = ops.atan(p.z, p.x) # -np.pi to np.pi
            r = ops.length(p.xz)
            _nrep = self.nrep / 2.0
            spacing = np.pi / _nrep
            theta_prime = theta - spacing * ops.clamp(ops.round(theta / spacing), -_nrep, _nrep)
            ty = ops.sin(theta_prime) * r
            tx = ops.cos(theta_prime) * r
            q = ops.vec3(tx, p.y, ty)
        elif self.axis == 'z':
            theta = ops.atan(p.y, p.x) # -np.pi to np.pi
            r = ops.length(p.xy)
            _nrep = self.nrep / 2.0
            spacing = np.pi / _nrep
            theta_prime = theta - spacing * ops.clamp(ops.round(theta / spacing), -_nrep, _nrep)
            ty = ops.sin(theta_prime) * r
            tx = ops.cos(theta_prime) * r
            q = ops.vec3(tx, ty, p.z)
        else: raise ValueError(f"Axis must be 'x' 'y' or 'z', not {self.axis}")
        return self.sdf(q)

class slice_2d(SDF):
    IS_2D = True
    def __init__(self, sdf: SDF, offset: float=0.0, axis: str = 'z'):
        self.offset = self.add_param(offset)
        self.axis = axis
        self.sdf = self.add_sdf(sdf)
        super().__init__()

    def opdef(self, p):
        match self.axis:
            case 'x': return self.sdf(ops.vec3(self.offset, p))
            case 'y': return self.sdf(ops.vec3(p.x, self.offset, p.y))
            case 'z': return self.sdf(ops.vec3(p, self.offset))
            case _: raise NotImplementedError(f'{self.axis}')
    
class extrude(SDF):
    IS_2D = False
    def __init__(self, sdf: SDF, height: float=0.0, axis='z', sym=False):
        assert sdf.is_2d == True
        assert axis in {'x', 'y', 'z'}
        self.height = self.add_param(height)
        self.axis = axis
        self.sdf = self.add_sdf(sdf)
        self.is_sym = sym
        super().__init__()

    def opdef(self, p):
        if self.axis == 'z':
            d = self.sdf(p.xy)
            q = ops.max(ops.vec2(d, ops.abs(p.z-self.height/2.0) - self.height/2.0), 0.0)
            q = ops.length(q) + ops.min(ops.max(d, ops.abs(p.z-self.height/2.0) - self.height/2.0), 0.0)
        elif self.axis == 'y':
            d = self.sdf(p.xz)
            q = ops.max(ops.vec2(d, ops.abs(p.y-self.height/2.0) - self.height/2.0), 0.0)
            q = ops.length(q) + ops.min(ops.max(d, ops.abs(p.y-self.height/2.0) - self.height/2.0), 0.0)
        else:
            d = self.sdf(p.yz)
            q = ops.max(ops.vec2(d, ops.abs(p.x-self.height/2.0) - self.height/2.0), 0.0)
            q = ops.length(q) + ops.min(ops.max(d, ops.abs(p.x-self.height/2.0) - self.height/2.0), 0.0)
        return q

class taper(SDF):
    def __init__(self, sdf, rate):
        self.sdf = self.add_sdf(sdf)
        self.rate = self.add_param(rate, DType(BaseType.float))
        super().__init__()

    def opdef(self, p):
        y_mod = p.y + p.x * self.rate * ops.sign(p.y)
        if self.sdf.is_2d: q = ops.vec2(p.x, y_mod)
        else: q = ops.vec3(p.x, y_mod, p.z)
        return self.sdf(q)

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
        q = ops.abs(p) - ops.vec3(self.x/2., self.y/2., self.z/2.)
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
        d: ops.param | float=0.0,
        r: ops.param | float=None,
    ):
        if r is not None: self.radius = self.add_param(r)
        else: self.diameter = self.add_param(d)
        super().__init__()

    def opdef(self, p):
        if hasattr(self, 'radius'): return ops.length(p) - self.radius
        else: return ops.length(p) - self.diameter / 2.0

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
        if hasattr(self, 'y_length'): q = ops.abs(p) - ops.vec2(self.x_length/2., self.y_length/2.)
        else: q = ops.abs(p) - self.x_length/2.
        return ops.length(ops.max(q, 0.0)) + ops.min(ops.max(q.x, q.y), 0.0)
    
class ngon_2d(SDF):
    IS_2D = True
    def __init__(self, sides: float, inscribed_diameter: float=1.0):
        self.sides = self.add_param(sides, dtype=DType(BaseType.float))
        self.inscribed_diameter = self.add_param(inscribed_diameter, dtype=DType(BaseType.float))
        super().__init__()
    
    def opdef(self, p):
        theta = ops.atan(p.x, -p.y)
        r = ops.length(p)
        rep_ang = 2.0*np.pi/self.sides
        theta = theta - ops.round(theta / rep_ang) * rep_ang

        # return ops.min(ops.cos(theta) * r - 0.5, 0.0) + ops.length(ops.vec2(ops.max(ops.max(ops.cos(theta+rep_ang) * r - 0.5, ops.cos(theta-rep_ang) * r - 0.5), 0.0), ops.max(ops.cos(theta) * r - 0.5, 0.0)))
        return ops.cos(theta) * r - self.inscribed_diameter/2.
    
class parabola(SDF):
    IS_2D = True
    def __init__(self, k: float):
        self.k_param = self.add_param(k, dtype=DType(BaseType.float))
        super().__init__()

    def opdef(self, p):
        pos = ops.vec2(ops.abs(p.x), p.y)
        k = self.k_param
        ik = 1.0/k
        p = ik*(pos.y - 0.5*ik)/3.0
        q = 0.25*ik*ik*pos.x
        h = q*q - p*p*p
        r1 = ops.pow(q+ops.sqrt(h),1.0/3.0)
        x1 = r1 + p/r1
        r2 = ops.sqrt(p)
        x2 = 2.0*r2*ops.cos(ops.acos(q/(p*r2))/3.0)
        x = ops.ternary(h>0.0, x1, x2)
        return ops.length(pos-ops.vec2(x,k*x*x)) * ops.sign(pos.x-x)
    
class hyperbola(SDF):
    IS_2D = True
    def __init__(self, k: float, he: float):
        self.k_param = self.add_param(k, dtype=DType(BaseType.float))
        self.he_param = self.add_param(he, dtype=DType(BaseType.float))
        super().__init__()

    def opdef(self, p):
        p = ops.abs(p)
        p = ops.vec2(p.x-p.y,p.x+p.y)/ops.sqrt(2.0)
        k = self.k_param
        he = self.he_param

        x2 = p.x*p.x/16.0
        y2 = p.y*p.y/16.0
        r = k*(4.0*k - p.x*p.y)/12.0
        q = (x2 - y2)*k*k
        h = q*q + r*r*r
        m1 = ops.sqrt(-r)
        u1 = m1*ops.cos( ops.acos(q/(r*m1))/3.0 )
        m2 = ops.pow(ops.sqrt(h)-q,1.0/3.0)
        u2 = (m2 - r/m2)/2.0
        u = ops.ternary(h<0.0, u1, u2)
        w = ops.sqrt( u + x2 )
        b = k*p.y - x2*p.x*2.0
        t = p.x/4.0 - w + ops.sqrt( 2.0*x2 - u + b/w/4.0 )
        t = ops.max(t,ops.sqrt(he*he*0.5+k)-he/ops.sqrt(2.0))
        d = ops.length( p-ops.vec2(t,k/t) )
        return ops.ternary(p.x*p.y < k, d, -d)
    
class shell(SDF):
    def __init__(self, sdf: SDF):
        self.sdf = self.add_sdf(sdf)
        super().__init__()
    
    def opdef(self, p):
        return ops.abs(self.sdf(p))
    
class twist(SDF):
    def __init__(self, sdf, units_per_turn = 1.0):
        self.sdf = self.add_sdf(sdf)
        self.units_per_turn = self.add_param(units_per_turn)
        super().__init__()

    def opdef(self, p):
        r = ops.length(p.xy)
        theta = ops.atan(p.y, p.x) - p.z * (np.pi * 2.0) / self.units_per_turn
        q = ops.vec3(ops.cos(theta) * r, ops.sin(theta) * r, p.z)
        return self.sdf(q)
    
class equilateral(SDF):
    IS_2D = True
    def __init__(self, diameter):
        self.diameter = self.add_param(diameter)
        super().__init__()

    def opdef(self, p):
        r = self.diameter/2.0
        k = ops.sqrt(3.0)
        px = ops.abs(p.x) - r
        py = p.y + r/k
        p = ops.ternary(px+k*py>0.0, ops.vec2(px-k*py,-k*px-py)/2.0, ops.vec2(px, py))
        px = p.x - ops.clamp( p.x, -2.0*r, 0.0 )
        p = ops.vec2(px, p.y)
        return -ops.length(p)*ops.sign(p.y)
    
# class slice(SDF):
#     IS_2D = False
#     def __init__(self, sdf, offset, axis='x'):
#         self.sdf = self.add_sdf(sdf)
#         self.offset = self.add_param(offset, dtype=DType(BaseType.float))
#         self.axis = axis
#         assert axis in {'x', 'y', 'z'}
#         super().__init__()
    
#     def opdef(self, p):
#         if self.axis == 'x':
#             q = p.x - self.offset
#             return ops.length(ops.vec2(self.sdf(ops.vec3(ops.min(q, 0.0), p.yz)), ops.max(q, 0.0)))
#         elif self.axis == 'y':
#             q = p.y - self.offset
#             return ops.length(ops.vec2(self.sdf(ops.vec3(p.x, ops.min(q, 0.0), p.z)), ops.max(q, 0.0)))
#         else:
#             q = p.z - self.offset
#             s = self.sdf(ops.vec3(p.xy, ops.min(q, 0.0)))
#             return s + ops.length(ops.vec2(s, ops.max(q, 0.0)))

