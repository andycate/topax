import numpy as np
from topax.sdfs import SDF, ngon_2d, circle, box, sphere
import topax.ops as ops
from topax.types import DType, BaseType
from topax import show_part

class frzcurvetile(SDF):
    IS_2D = False
    def __init__(self, R_n, Z_n, n_reps, unit: SDF, sym=True):
        assert not unit.is_2d
        assert sym, "Only sym=True is currently supported"
        self.sdf = self.add_sdf(unit)
        self.R0 = self.add_param([R_n[4], 0.0], dtype=DType(BaseType.vec2))
        self.Z0 = self.add_param([0.0, Z_n[4]], dtype=DType(BaseType.vec2))
        self.axis_R_cos_1to4 = self.add_param(R_n[5:9], dtype=DType(BaseType.vec4))
        self.axis_R_sin_1to4 = self.add_param(R_n[0:4][::-1], dtype=DType(BaseType.vec4))
        self.axis_Z_cos_1to4 = self.add_param(Z_n[5:9], dtype=DType(BaseType.vec4))
        self.axis_Z_sin_1to4 = self.add_param(Z_n[0:4][::-1], dtype=DType(BaseType.vec4))
        self.n_reps = self.add_param(n_reps, DType(BaseType.float))
        super().__init__()

    def opdef(self, p):
        zeta = ops.atan(p.y, p.x)
        spacing = np.pi * 2. / self.n_reps
        zeta_quant = ops.round(zeta / spacing) * spacing
        zeta_cos_1to4 = ops.cos(ops.vec4(zeta_quant, 2.0 * zeta_quant, 3.0 * zeta_quant, 4.0 * zeta_quant) * 2.0)
        zeta_sin_1to4 = ops.sin(ops.vec4(zeta_quant, 2.0 * zeta_quant, 3.0 * zeta_quant, 4.0 * zeta_quant) * 2.0)
        r0 = self.R0 + self.Z0
        r0 = ops.vec2(
            r0.x + ops.dot(zeta_cos_1to4, self.axis_R_cos_1to4) + ops.dot(zeta_sin_1to4, self.axis_R_sin_1to4),
            r0.y + ops.dot(zeta_cos_1to4, self.axis_Z_cos_1to4) + ops.dot(zeta_sin_1to4, self.axis_Z_sin_1to4)
        )
        r = ops.vec2(ops.length(p.xy), p.z) - r0

        xy_quant = ops.vec2(ops.cos(zeta_quant), ops.sin(zeta_quant))
        qz = ops.length(p.xy - ops.dot(xy_quant, p.xy) * xy_quant)

        return self.sdf(ops.vec3(r, qz))
    

R_n = [ 1.06189048e-03, -5.65066293e-03, -1.67871897e-02, -1.83663048e-01,
  1.00800650e+00,  1.22565200e-01,  5.74603040e-03, -1.36255935e-03,
 -2.66876307e-04]
Z_n = [ 3.40067618e-04,  1.20266170e-03, -1.87913889e-03, -1.09281308e-01,
  -0.24499754000411247, -1.33612715e-01, -2.21045784e-02, -2.94654958e-03,
 -1.16186449e-04]

rot_z = ops.param(None, 'rot_z', 0., vmin=0., vmax=360.)

show_part(
    frzcurvetile(R_n, Z_n, 280., box(0.01).tlp(0.025, 5.).r(rot_z, 'z')),
    [0.1, 0.2, 0.6]
)



# n_sides = ops.param(DType(BaseType.float), 'n_sides', 6., vmin=2, vmax=12)
# id_m = ops.param(DType(BaseType.float), 'id_m', 0.08, vmax=0.2)
# od_m = ops.param(DType(BaseType.float), 'od_m', 0.1, vmax=0.2)

# show_part(
#     ngon_2d(n_sides, od_m).sub(circle(r=id_m)),
#     [0.1, 0.2, 0.5]
# )

