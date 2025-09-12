import abc

import jax.numpy as jnp

class OpBase(abc.ABC):
    pass

# class OpConst(OpBase):
#     def __init__(self, jax_const, glsl_const):
#         self.jax_const = jax_const
#         self.glsl_const = glsl_const

#     def glsl_value(self):
#         return self.glsl_const
    
# inf = OpConst(jnp.inf, "uintBitsToFloat(0x7F800000u)")


class add(OpBase):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

table = {
    "add(0.3,0.4)": add(0.3, 0.4),
    "add(add(0.3,0.4),0.4)": add()
}


class subtract(OpBase):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
