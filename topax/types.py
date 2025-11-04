from enum import Enum
from dataclasses import dataclass

class BaseType(Enum):
    float = 0
    int = 1
    bool = 2
    vec2 = 3
    vec3 = 4
    vec4 = 5
    mat2 = 6
    mat3 = 7
    mat4 = 8

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class DType:
    base: BaseType
    length: int = None

    def get_shader_declaration(self, name):
        return f"{self.base.name} {name}" + (f"[{self.length}]" if self.length != None else "")

def get_singular_value(value):
    if hasattr(value, '__iter__'): return get_singular_value(value[0])
    else: return value

def get_shape(value):
    if hasattr(value, 'shape'): return value.shape
    s = []
    v = value
    while hasattr(v, '__iter__'):
        s.append(len(v))
        v = v[0]
    return tuple(s)

def resolve_singular_base_type(value):
    if isinstance(value, float): return BaseType.float
    if isinstance(value, int): return BaseType.int
    if hasattr(value, 'dtype'):
        raw_type_name = value.dtype.name
        if raw_type_name.find('float') > -1: return BaseType.float
        if raw_type_name.find('int') > -1: return BaseType.int
    raise ValueError(f"Can't resolve type of value {value}")

def resolve_dtype(value):
    singular_value = get_singular_value(value)
    singular_type = resolve_singular_base_type(singular_value)
    shape = get_shape(value)
    if len(shape) == 0: return DType(singular_type)
    if len(shape) == 1:
        assert singular_type == BaseType.float
        match shape[0]:
            case 2: return DType(BaseType.vec2)
            case 3: return DType(BaseType.vec3)
            case 4: return DType(BaseType.vec4)
            case _: pass
    if len(shape) == 2:
        assert singular_type == BaseType.float
        assert shape[0] == shape[1]
        match shape[0]:
            case 2: return DType(BaseType.mat2)
            case 3: return DType(BaseType.mat3)
            case 4: return DType(BaseType.mat4)
            case _: pass
    raise NotImplementedError(f"value type cannot be resolved for {value}")
    