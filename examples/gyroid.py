from topax import show_part
from topax.sdfs_old import SDF, sphere, box, intersect, gyroid, offset
import topax.ops_old as ops_old
from topax.ops_old import DType


show_part(
    intersect(
        offset(gyroid(1.0, 0.08, 0.33), 0.2),
        offset(box(1.0), 0.2)
    ),
    [0.2, 0.4, 0.6]
)
