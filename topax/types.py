from enum import Enum
from dataclasses import dataclass

class TypeEnum(Enum):
    float = 1
    int = 2
    ivec2 = 3
    ivec3 = 4
    ivec4 = 5
    vec2 = 6
    vec3 = 7
    vec4 = 8
    mat2 = 9
    mat3 = 10
    mat4 = 11

    def __repr__(self):
        return self.name

@dataclass(frozen=True)
class DType:
    type: TypeEnum

    def get_shader_declaration(self, name: str):
        return self.type.name + ' ' + name

@dataclass(frozen=True)
class ArrayType(DType):
    length: int

    def get_shader_declaration(self, name: str):
        return self.type.name + ' ' + name + f'[{self.length}]'

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

def resolve_singular_type(value):
    if isinstance(value, float): return TypeEnum.float
    if isinstance(value, int): return TypeEnum.int
    if hasattr(value, 'dtype'):
        raw_type_name = value.dtype.name
        if raw_type_name.find('float') > -1: return TypeEnum.float
        if raw_type_name.find('int') > -1: return TypeEnum.int
        return None

def resolve_type(value):
    singular_value = get_singular_value(value)
    singular_type = resolve_singular_type(singular_value)
    if singular_type is None: raise ValueError(f"Can't resolve type of value {value}")
    shape = get_shape(value)
    if len(shape) == 0: return DType(singular_type)
    if len(shape) == 1:
        matches = {
            (TypeEnum.float, (2,)): DType(TypeEnum.vec2),
            (TypeEnum.float, (3,)): DType(TypeEnum.vec3),
            (TypeEnum.float, (4,)): DType(TypeEnum.vec4),
            (TypeEnum.int, (2,)): DType(TypeEnum.ivec2),
            (TypeEnum.int, (3,)): DType(TypeEnum.ivec3),
            (TypeEnum.int, (4,)): DType(TypeEnum.ivec4),
            (TypeEnum.float, (2,2)): DType(TypeEnum.mat2),
            (TypeEnum.float, (3,3)): DType(TypeEnum.mat3),
            (TypeEnum.float, (4,4)): DType(TypeEnum.mat4),
        }
        if (singular_type, shape) not in matches:
            raise ValueError(f"Can't resolve type of value {value}")
        return matches[(singular_type, shape)]
    