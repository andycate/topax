import warnings
import abc
from dataclasses import dataclass
import inspect
from typing import Any

import jax
import jax.numpy as jnp

@dataclass
class ShaderParam:
    name: str
    type: str
    value: Any = None

@dataclass
class ShaderMethod:
    params: list[ShaderParam]
    contents: str
    name: str = ""

class SDF(abc.ABC):
    _all_sdfs = {}

    def __init__(self, *args):
        for i, s in enumerate(args):
            assert type(s).__base__ == SDF, f"unsupported hashable type {type(s)} detected, SDFs currently only support other SDFs as hashable types"
            assert s not in args[i+1:], "Currently, passing multiple of the same sdf instance to a modifier sdf is not supported"
        self._child_sdfs = args
        
        i = 0
        while True:
            if not i in SDF._all_sdfs[type(self).__name__]:
                SDF._all_sdfs[type(self).__name__].add(i)
                self._inst_name = type(self).__name__ + f"{i}"
                break
            i += 1

    def __del__(self):
        try:
            idx = int(self._inst_name.removeprefix(type(self).__name__))
            if type(self).__name__ in SDF._all_sdfs:
                SDF._all_sdfs[type(self).__name__].discard(idx)
        except Exception:
            # don't raise from __del__ — it's unsafe and can crash interpreter shutdown
            pass

    def __init_subclass__(cls) -> None:
        if cls.__name__ in SDF._all_sdfs:
            raise ValueError(f"duplicate SDF class called {cls.__name__} cannot be created since one with that name already exists!")
        SDF._all_sdfs[cls.__name__] = set()
        cls._child_sdfs = []
        cls._param_names = {}
        cls._is_modifier = False
        for _, (param_name, param_obj) in enumerate(inspect.signature(cls.__init__).parameters.items()):
            if param_name == 'self':
                continue
            if param_obj.annotation == SDF:
                cls._child_sdfs.append(param_name)
                cls._is_modifier = True
            else:
                assert param_obj.annotation == 'vec3' or param_obj.annotation == float, f"invalid sdf parameter '{param_name}' of type {param_obj.annotation} found for sdf {cls.__name__}"
                cls._param_names[param_name] = param_obj.annotation if type(param_obj.annotation) == str else param_obj.annotation.__name__

        if not cls._is_modifier:
            assert hasattr(cls, "sdf_definition")

    def get_all_sdf_types(self, sdf_types: set):
        for s in self._child_sdfs:
            s.get_all_sdf_types(sdf_types)
            sdf_types.add(type(s))
        sdf_types.add(type(self))

    def get_resolved_param_names(self):
        pnames = {}
        for k in type(self)._param_names:
            pnames[k] = f"sdfin_{self._inst_name}_{k}"
        return pnames

    def get_all_param_names(self):
        pn = dict()
        for s in self._child_sdfs:
            pn.update(s.get_all_param_names())
        resolved = self.get_resolved_param_names()
        for k in resolved:
            pn[resolved[k]] = type(self)._param_names[k]
        return pn
    
    def get_all_param_values(self):
        values = dict()
        for s in self._child_sdfs:
            values.update(s.get_all_param_values())
        resolved = self.get_resolved_param_names()
        for pn in resolved:
            attribute = self.__getattribute__(pn)
            if hasattr(attribute, "copy"):
                attribute = attribute.copy()
            values[pn] = ShaderParam(name=resolved[pn], type=type(self)._param_names[pn], value=attribute)
        return values

    def generate_shader(self, template):
        sdf_types = set()
        funcs = []
        self.get_all_sdf_types(sdf_types)
        for st in sdf_types:
            if not st._is_modifier:
                funcs.append(
                    ShaderMethod(
                        name=st.__name__,
                        params=[ShaderParam(name=k, type=st._param_names[k]) for k in st._param_names],
                        contents=st.sdf_definition()
                    )
                )
        cd = self.calling_definition("p", "d")
        pnames = self.get_all_param_names()
        return template.render(inputs=[dict(name=p, type=pnames[p]) for p in pnames], funcs=funcs, main_sdf=cd)

    @abc.abstractmethod
    def jax_definition(self, p):
        """
        This method performs some jax operations
        """
        return jnp.inf
    
    def calling_definition(self, p: str, ret: str):
        """
        Return the block of shader code that goes in main and calls this SDF instance.

        :param p: string representing input point to this SDF
        :param ret: string of variable to write to
        """
        if not type(self)._is_modifier:
            return f"{ret} = sdf_{type(self).__name__}({p + (", " if len(type(self)._param_names) > 0 else "") + ", ".join([f"sdfin_{self._inst_name}_{k}" for k in type(self)._param_names])});"
        else:
            raise NotImplementedError()


    def get_hash(self):
        if not hasattr(self, "_child_sdfs"):
            warnings.warn(f"sdf class {type(self).__name__} has no hashable items; did you forget to call super() constructor?")
            return f"{type(self).__name__}()"
        assert len(type(self)._child_sdfs) == len(self._child_sdfs), "number of hashable items reported and detected in SDF signature don't match!"
        hashs = []
        for h in self._child_sdfs:
            hashs.append(h.get_hash())
        return f"{type(self).__name__}(" + ",".join(hashs) + ")"
    
    def __call__(self, p):
        return self.jax_definition(p)

class empty(SDF):
    def __init__(self):
        super().__init__()
    
    def jax_definition(self, p):
        return jnp.inf
    
    @classmethod
    def sdf_definition(cls):
        return "return POS_INFINITY;"

class translate(SDF):
    def __init__(self, sdf_in: SDF, offset: 'vec3'):
        super().__init__(sdf_in)
        self.sdf_in = sdf_in
        self.offset = offset
    
    def jax_definition(self, p):
        offset = jnp.atleast_1d(self.offset)
        return self.sdf_in(p - offset)
    
    def calling_definition(self, p: str, ret: str):
        params = self.get_resolved_param_names()
        return self.sdf_in.calling_definition(f"{p} - {params['offset']}", ret)
    
# class union(SDF):
#     def __init__(self, *sdfs_in: SDF):
#         super().__init__(*sdfs_in)
#         self._sdfs = sdfs_in
    
#     def jax_definition(self, p):
#         dists = jnp.zeros(len(self._sdfs))
#         for i, s in enumerate(self._sdfs):
#             dists = dists.at[i].set(s(p))
#         return jnp.min(dists).item()

class sphere(SDF):
    def __init__(self, radius: float, center: 'vec3'=[0.0, 0.0, 0.0]):
        super().__init__()
        self.radius = radius
        self.center = center
    
    def jax_definition(self, p):
        center = jnp.atleast_1d(self.center)
        return jnp.linalg.norm(p - center) - self.radius
    
    @classmethod
    def sdf_definition(cls):
        return "return length(p - center) - radius;"


# def sphere(radius, position=[0., 0., 0.]):
#     position = jnp.atleast_2d(position)
#     def _sphere(p):
#         p = jnp.atleast_2d(p)
#         return norm(p - position) - radius
#     return _sphere

# def box(dims, position=[0., 0., 0.]):
#     s = jnp.atleast_2d(dims)
#     position = jnp.atleast_2d(position)
#     def _box(p):
#         d = jnp.abs(p-position)-s
#         return jnp.linalg.norm(jnp.maximum(d,0.0), axis=-1) + jnp.minimum(jnp.max(d, axis=-1),0.0)
#     return _box

# def cylinder(radius, height):
#     radh = jnp.array([[radius, height]])
#     def _cylinder(p):
#         d = jnp.abs(jnp.c_[jnp.linalg.norm(p[:,:2], axis=-1),p[:,2]]) - radh
#         return jnp.minimum(jnp.max(d, axis=-1),0.0) + jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
#     return _cylinder

# # def capped_cylinder(a, b, r):
# #     a = jnp.atleast_2d(a)
# #     b = jnp.atleast_2d(b)
# #     ba = b - a
# #     baba = (ba * ba).sum(axis=-1)
# #     def _capped_cylinder(p):
# #         pa = p - a
# #         paba = (pa * ba).sum(axis=-1)
# #         x = jnp.linalg.norm(pa*baba-ba*paba, axis=-1) - r*baba
# #         y = jnp.abs(paba-baba*0.5)-baba*0.5
# #         x2 = x*x
# #         y2 = y*y*baba
# #         # d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
# #         # return sign(d)*sqrt(abs(d))/baba
# #     return _capped_cylinder

# def translate(f, d):
#     d = jnp.atleast_2d(d)
#     def _translate(p):
#         return f(p - d)
#     return _translate

# def scale(f, factor):
#     def _scale(p):
#         return f(p/factor) * factor
#     return _scale

# def union(*args):
#     def _union(p):
#         distances = jnp.zeros((p.shape[0], len(args)))
#         for i, f in enumerate(args):
#             distances = distances.at[:, i].set(f(p))
#         return jnp.min(distances, axis=-1)
#     return _union

# def intersect(*args):
#     def _intersect(p):
#         distances = jnp.zeros((p.shape[0], len(args)))
#         for i, f in enumerate(args):
#             distances = distances.at[:, i].set(f(p))
#         return jnp.max(distances, axis=-1)
#     return _intersect

# def subtract(f, tool):
#     def _subtract(p):
#         return jnp.maximum(-tool(p), f(p))
#     return _subtract

# # def circular_pattern(f, cnt):
# #     rotations = jnp.zeros((cnt, 3, 3))
# #     for i in range(cnt):
# #         rotations = rotations.at[i].set(rotation_matrix(-i * 2 * jnp.pi / cnt, 'y'))
# #     def _circular_pattern(p):
# #         p_mod = rotations @ p
# #         return jnp.min(jax.vmap(f)(p_mod))
# #     return _circular_pattern

# def test_sdf():
#     part1 = union(
#         sphere(0.5),
#         box([0.3, 0.3, 0.3], position=[0.5, 0.5, 0.5]),
#     )
#     return part1
