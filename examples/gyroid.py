from topax import show_part
from topax.sdfs import SDF, sphere, box, intersect, gyroid, offset, shell
import topax.ops as ops
from topax.ops import DType, BaseType


slice_offset = ops.param(None, 'slice_offset', 0.0, vmin=-2.0, vmax=2.0)


show_part(
    gyroid(1.0, 0.08, 0.33).s(0.1).i(sphere(2.0)).u(shell(sphere(2.0)).o(0.1)).sub(box(10.).t(y=-5.+slice_offset)),
    [0.0, 0.1, 0.2]
)
