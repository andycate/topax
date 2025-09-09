import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Union

import warnings
import abc
from dataclasses import dataclass
import inspect

@dataclass
class ShaderParam:
    name: str
    type: str

@dataclass
class ShaderMethod:
    params: list[ShaderParam]
    contents: str
    name: str = ""

class SDF(abc.ABC):
    _all_sdfs = {}

    def __init__(self, *args):
        for s in args:
            assert type(s).__base__ == SDF, f"unsupported hashable type {type(s)} detected, SDFs currently only support other SDFs as hashable types"
        self._inner_sdfs = args
        self._hashable_items = args
        self._name = type(self).__name__ + f"{SDF._all_sdfs[type(self).__name__]}"
        SDF._all_sdfs[type(self).__name__] += 1

    def __init_subclass__(cls) -> None:
        if cls.__name__ in SDF._all_sdfs:
            raise ValueError(f"duplicate SDF class called {cls.__name__} cannot be created since one with that name already exists!")
        SDF._all_sdfs[cls.__name__] = 0
        cls._hashable_items = []
        cls._param_names = {}
        cls._is_modifier = False
        for _, (param_name, param_obj) in enumerate(inspect.signature(cls.__init__).parameters.items()):
            if param_name == 'self':
                continue
            if param_obj.annotation == SDF:
                cls._hashable_items.append(param_name)
                cls._is_modifier = True
            else:
                assert param_obj.annotation == 'vec3' or param_obj.annotation == float, f"invalid sdf parameter '{param_name}' of type {param_obj.annotation} found for sdf {cls.__name__}"
                cls._param_names[param_name] = param_obj.annotation if type(param_obj.annotation) == str else param_obj.annotation.__name__

        if not cls._is_modifier:
            assert hasattr(cls, "shader_definition")

        # def make_init_wrapper(init_func, _hashable_items):
        #     def init_wrapper(self, *args, **kwargs):
        #         self._hashable_items = []
        #         for i, name in _hashable_items:
        #             if len(args) > i:
        #                 self._hashable_items.append(args[i])
        #             else:
        #                 self._hashable_items.append(kwargs[name])
        #         return cls.__init__(self, *args, **kwargs)
        #     return init_wrapper
        # cls.__init__ = make_init_wrapper(cls.__init__, _hashable_items)

        # print(cls.__init__.__annotations__)
        # print(inspect.signature(cls.__init__))

    def get_all_sdf_types(self, sdf_types: set):
        for s in self._inner_sdfs:
            s.get_all_sdf_types(sdf_types)
            sdf_types.add(type(s))
        sdf_types.add(type(self))

    def generate_shader(self):
        sdf_types = set()
        funcs = []
        self.get_all_sdf_types(sdf_types)
        for st in sdf_types:
            if not st._is_modifier:
                funcs.append(
                    ShaderMethod(
                        name=st.__name__,
                        params=[ShaderParam(name=k, type=st._param_names[k]) for k in st._param_names],
                        contents=st.shader_definition()
                    )
                )
        return funcs

    @abc.abstractmethod
    def jax_definition(self, p):
        """
        This method performs some jax 
        """
        return jnp.inf

    def get_hash(self):
        if not hasattr(self, "_hashable_items"):
            warnings.warn(f"sdf class {type(self).__name__} has no hashable items; did you forget to call super() constructor?")
            return f"{type(self).__name__}()"
        assert len(type(self)._hashable_items) == len(self._hashable_items), "number of hashable items reported and detected in SDF signature don't match!"
        hashs = []
        for h in self._hashable_items:
            if not hasattr(h, "get_hash"):
                hashs.append(h.__hash__())
            else:
                hashs.append(h.get_hash())
        return f"{type(self).__name__}(" + ",".join(hashs) + ")"
    
    def __call__(self, p):
        return self.jax_definition(p)

class translate(SDF):
    def __init__(self, sdf_in: SDF, offset: 'vec3'):
        super().__init__(sdf_in)
        self._sdf = sdf_in
        self._offset = offset
    
    def jax_definition(self, p):
        offset = jnp.atleast_1d(self._offset)
        return self._sdf(p - offset)
    
    # def sdft_definition(self)
    
    # @staticmethod
    # def shader_definition():
    #     return ShaderMethod(
    #         params=[ShaderParam("offset", "vec3")],
    #         contents="return p - offset;"
    #     )
    
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
        self._radius = radius
        self._center = center
    
    def jax_definition(self, p):
        center = jnp.atleast_1d(self._center)
        return jnp.linalg.norm(p - center) - self._radius
    
    @classmethod
    def shader_definition(cls):
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
