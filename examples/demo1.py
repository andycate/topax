import numpy as np
from topax import show_part
import topax.ops as ops
from topax.sdfs import SDF, shell, circle, rectangle, parabola, hyperbola, extrude, ngon_2d
from topax.types import DType, BaseType

diameter_od = ops.param(None, 'diameter_od', 1.0, vmin=0.1, vmax=5.0)
ring_thickness = ops.param(None, 'ring_thickness', 0.1, vmin=0.01, vmax=1.0)
hole_offset = ops.param(None, 'hole_offset', 0.0, vmin=-1.0, vmax=1.0)
hole_diameter = ops.param(None, 'hole_diameter', 0.1, vmin=0.0, vmax=.5)
pattern_radius = ops.param(None, 'pattern_radius', 0.35, vmin=0.0, vmax=2.0)
n_reps = ops.param(None, 'n_reps', 6., resolution=1., vmin=1., vmax=100.)
offset = ops.param(None, 'offset', 0.0, vmin=-0.5, vmax=1.)
parabola_k = ops.param(None, 'parabola_k', 1.0, vmin=0.1, vmax=5.0)
hyperbola_k = ops.param(None, 'hyperbola_k', 0.03, vmin=0.001, vmax=0.6)
hyperbola_he = ops.param(None, 'hyperbola_he', 0.5, vmin=0.1, vmax=3.0)
inscribed_diameter = ops.param(None, 'inscribed_diameter', 1.0, vmin=0.1, vmax=5.0)
n_sides = ops.param(None, 'n_sides', 3., vmin=3., vmax=24.)


show_part(
    # 1. A simple circle with adjustable diameter.
    #    The colors show the SDF values everywhere in space.
    # circle(diameter_od),

    # 2. A ring with adjustable outer diameter and thickness.
    # circle(diameter_od).sub(
    #     circle(diameter_od - ring_thickness * 2.0)
    # ),

    # 3. A circle with adjustable outer diameter and thickness.
    # shell(circle(diameter_od)),

    # 4. A ring with adjustable outer diameter and thickness.
    # shell(circle(diameter_od)).o(ring_thickness/2.),

    # 5. A circle with a hole offset from center. Notice how when the cutout gets
    #    close to the edge, the model doesn't break!
    # circle(diameter_od).sub(
    #     circle(diameter_od - ring_thickness * 2.0).t(y=hole_offset)
    # ),

    # 6. A circular pattern of holes cut out of a disk. Again, the model is robust
    #    even when the holes get close to the edge.
    # circle(diameter_od).sub(
    #     circle(hole_diameter).t(x=pattern_radius).cp(n_reps)
    # ),

    # 7. A square. Notice how contours are rounded on the edges!
    #.   Notice how when I offset the boundary outward, the edges are rounded!!!!
    # rectangle(1.0).o(offset),

    # 7. An ngon. Notice how contours are rounded on the edges!
    #.   Notice how when I offset the boundary outward, the edges are rounded!!!!
    # ngon_2d(n_sides, inscribed_diameter).o(offset),

    # 8. A parabolic shape with adjustable offset.
    # parabola(parabola_k).t(y=-offset),

    # 9. A hyperbolic shape with adjustable offset.
    hyperbola(hyperbola_k, hyperbola_he),
    [0.1, 0.2, 0.7]
)

# show_part(
#     circle(inscribed_diameter),
#     [0.7, 0.2, 0.1]
# )
