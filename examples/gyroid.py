from topax import show_part
from topax.sdfs import SDF, sphere, box, intersect
import topax.ops as ops
from topax.ops import DType

class gyroid(SDF):
    def __init__(self, scale: float, fill: float):
        self.add_input('scale', scale, DType.float)
        self.add_input('fill', fill, DType.float)

    def sdf_definition(self, p):
        scaled_p = p * self.scale
        gyroid = ops.abs(ops.dot(ops.sin(scaled_p), ops.cos(scaled_p.yzx))) * 0.33 - self.fill
        return gyroid

show_part(
    intersect(
        gyroid(2.0, 0.05),
        box(5.0)
    ),
    [0.2, 0.4, 0.6]
)
