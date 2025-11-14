import numpy as np
from topax import show_part
import topax.ops as ops
from topax.sdfs import SDF, rectangle, box, cylinder, twist, extrude, ngon_2d, equilateral, sphere, gyroid, shell, slice_2d
from topax.types import DType, BaseType

class thread(SDF):
    IS_2D = False
    def __init__(self, profile, units_per_turn = 1.0, number_repeats = 1.0):
        self.profile_sdf = self.add_sdf(profile)
        self.units_per_turn = self.add_param(units_per_turn)
        self.number_repeats = self.add_param(number_repeats)
        super().__init__()

    def opdef(self, p):
        theta = ops.atan(p.y, p.x) * 0.5 / (2.0 * np.pi)
        r = ops.length(p.xy)
        z = p.z/self.units_per_turn + theta*self.number_repeats
        z = z - ops.round(z / 0.5) * 0.5
        return self.profile_sdf(ops.vec2(r, z*self.units_per_turn))


side_length = ops.param(None, 'side_length', 1.0, vmin=0.1, vmax=5.0)
height = ops.param(None, 'height', 0.5, vmin=0.1, vmax=5.0)
x_offset = ops.param(None, 'x_offset', 0.0, vmin=-5.0, vmax=5.0)
z_angle = ops.param(None, 'z_angle', 0.0, vmin=-180.0, vmax=180.0)
fillet_radius = ops.param(None, 'fillet_radius', 0.0, vmin=0.0, vmax=1.)
hole_diameter = ops.param(None, 'hole_diameter', 0.2, vmin=0.0, vmax=2.0)
twist_rate = ops.param(None, 'twist_rate', 1.0, vmin=0.1, vmax=5.0)
thread_length = ops.param(None, 'thread_length', 1.0, vmin=0.1, vmax=5.0)
thread_pitch = ops.param(None, 'thread_pitch', 0.25, vmin=0.01, vmax=0.5)
num_bolt_repeats = ops.param(None, 'num_bolt_repeats', 5.0, resolution=1., vmin=1.0, vmax=30.0)
gyroid_scale = ops.param(None, 'gyroid_scale', 0.2, vmin=0.05, vmax=1.0)




show_part(
    # 1. We can extrude 2D shapes just like parametric CAD!
    # rectangle(side_length).extrude(height, axis='z', sym=True),

    # 2. We can just make a box directly. What if we want to rotate it/translate it?
    #    How about filleting the edges?
    # box(
    #     x=side_length-fillet_radius*2., 
    #     y=side_length-fillet_radius*2., 
    #     z=height-fillet_radius*2.
    # ).o(fillet_radius).sub(cylinder(hole_diameter, height+0.1).r(90.)).t(x=x_offset).r(z_angle, 'z'),

    # 3. Twisting!
    # twist(
    #     box(
    #         x=side_length-fillet_radius*2., 
    #         y=side_length-fillet_radius*2., 
    #         z=height-fillet_radius*2.
    #     ).o(fillet_radius).sub(cylinder(hole_diameter, height+0.1).r(90.)).t(x=x_offset).r(z_angle, 'z'),
    # twist_rate),

    #4. Threads!,
    # thread(
    #     equilateral(0.1).r(90.).sub(
    #         rectangle(0.2, 1.0).t(x=0.125)
    #     ).t(x=0.225),
    #     units_per_turn=thread_pitch,
    # ).u(cylinder(0.205, 1.5).r(90.0)).i(box(x=2., y=2., z=thread_length)).u(
    #     extrude(ngon_2d(6, 0.75), 0.25).t(z=thread_length/2.)
    # ),#.t(x=2.).cp(num_bolt_repeats, axis='z'),

    # 5. Mesh structures!
    # gyroid().s(gyroid_scale).i(sphere(side_length)),#.u(shell(sphere(side_length)).o(0.05)).sub(box(30.).t(y=-15.+x_offset)),
    # gyroid().r(z_angle)),
    # gyroid().s(gyroid_scale).i(box(side_length)).tlp(2.2, 5., 'x').tlp(2.2, 5., 'y').tlp(2.2, 5., 'z'),
    # box(side_length).o(0.1).tlp(2.2, 50., 'x').tlp(2.2, 50., 'y').tlp(2.2, 50., 'z'),

    


    # 4. Mesh structures!
    [0.1, 0.2, 0.7]
)
