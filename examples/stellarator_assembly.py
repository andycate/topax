import numpy as np

from examples.sdf_extensions import frzsurface, fxycoil, frzcurvesweep, frzcurvetile
from topax import show_part
import topax.ops as ops
from topax.sdfs import slice_2d, rectangle, SDF, shell, box, gyroid
from topax._utils import normalize
from topax.types import DType, BaseType

import desc.examples
from desc.grid import LinearGrid
from desc.geometry import FourierRZCurve
from desc.coils import FourierXYCoil

eq = desc.examples.get("precise_QA")

show_part(
    frzsurface(eq),#.sub(box(4.).t(y=-2.)),
    [0.3, 0.0, 0.3]
)

eq_off = eq.surface.constant_offset_surface(0.2, grid=LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP))
show_part(
    shell(frzsurface(eq, eq_off).o(0.0)).sub(box(4.).t(y=-2.)),
    [0.2, 0.2, 0.2]
)

test_coil = FourierXYCoil(1.0, [0., 0., 0.], normalize(np.array([0., 0., 1.])), [0.02, 0.0, 0.5, 0., 0.0, 0.0], [0.0, 0.0, 0.0, -0.8, 0.0, 0.02], modes=[-3, -2, -1, 1, 2, 3])
test_coil = fxycoil(test_coil, 16, rectangle(0.08, 0.05))



show_part(
    test_coil.r(105.).t(x=1.2).i(gyroid().s(0.01)),
    [0.5, 0.3, 0.07]
)

show_part(
    test_coil.r(105.).r(-65., 'y').r(-25., 'z').t(x=1., y=0.5, z=-0.1),
    [0.5, 0.3, 0.07]
)

show_part(
    test_coil.r(105.).r(-95., 'y').r(-65., 'z').t(x=0.3, y=0.8, z=-0.1),
    [0.5, 0.3, 0.07]
)

show_part(
    test_coil.r(105.).t(x=1.2).r(180., 'y'),
    [0.5, 0.3, 0.07]
)

show_part(
    test_coil.r(105.).r(-65., 'y').r(-25., 'z').t(x=1., y=0.5, z=-0.1).r(180., 'y'),
    [0.5, 0.3, 0.07]
)

show_part(
    test_coil.r(105.).r(-95., 'y').r(-65., 'z').t(x=0.3, y=0.8, z=-0.1).r(180., 'y'),
    [0.5, 0.3, 0.07]
)
    

R_n = [ 1.06189048e-03, -5.65066293e-03, -1.67871897e-02, -1.83663048e-01,
  1.00800650e+00,  1.22565200e-01,  5.74603040e-03, -1.36255935e-03,
 -2.66876307e-04]
Z_n = [ 3.40067618e-04,  1.20266170e-03, -1.87913889e-03, -1.09281308e-01,
  -0.24499754000411247, -1.33612715e-01, -2.21045784e-02, -2.94654958e-03,
 -1.16186449e-04]


angle = ops.param(None, 'angle', 45., vmin=0., vmax=180.)
spacing = ops.param(None, 'spacing', 0.058, vmin=0., vmax=0.15)
target_length = ops.param(None, 'target_length', 0.06, vmin=0.01, vmax=0.15)
z_pos = ops.param(None, 'z_pos', -0.095, vmin=-0.2, vmax=0.)

# show_part(
#     frzcurvesweep(R_n, Z_n, rectangle(0.01, 0.08)).t(z=z_pos),
#     [0.1, 0.2, 0.6]
# )
    
show_part(
    frzcurvesweep(R_n, Z_n, rectangle(0.01, target_length).r(angle).s(1./(1.0 + spacing))).s(1.0 + spacing).t(z=z_pos),
    [0.1, 0.2, 0.6]
)

show_part(
    frzcurvesweep(R_n, Z_n, rectangle(0.01, target_length).r(-angle).s(1./(1.0 - spacing))).s(1.0 - spacing).t(z=z_pos),
    [0.1, 0.2, 0.6]
)

show_part(
    frzcurvetile(R_n, Z_n, 260., box(0.02).i(gyroid().s(0.005)).tlp(0.025, 5.).t(y=0.0175).r(angle-90., 'z').s(1./(1.0 + spacing))).s(1.0 + spacing).t(z=z_pos),
    [0.05, 0.4, 0.1]
)

show_part(
    frzcurvetile(R_n, Z_n, 260., box(0.02).tlp(0.025, 5.).t(y=0.0175).r(-(angle-90.), 'z').s(1./(1.0 - spacing))).s(1.0 - spacing).t(z=z_pos),
    [0.05, 0.4, 0.1]
)
