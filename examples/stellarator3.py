"""
The goal of this example is to show how a DESC equilibrium can be
directly represented as an SDF!
"""

import numpy as np
import graphviz
from topax import show_part
from topax.sdfs_old import SDF, translate, rotate, union, subtract, intersect, box, cylinder, sphere
from topax.ops_old import Op, OpType, DType, sin, cos, vec2, vec4, atan, dot, length, exp2

from desc.geometry import FourierRZToroidalSurface
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.grid import LinearGrid, Grid
import desc.examples

eq = desc.examples.get("precise_QA")

class frzsurface(SDF):
    def __init__(self, surface: FourierRZToroidalSurface):
        super().__init__()
        self.surface = surface

        axis = eq.get_axis()

        N=eq.N*2+1
        M=eq.M*2+1

        sample_grid = LinearGrid(rho=[1.0], N=N, M=M, NFP=2)
        axis_points = axis.compute(["R", "Z"], grid=sample_grid)
        axis_points = np.c_[axis_points["R"], axis_points["Z"]]
        surface_points = surface.compute(["R", "Z"], grid=sample_grid)
        weights = surface.compute(["|e_theta x e_zeta|"], grid=sample_grid)["|e_theta x e_zeta|"]
        surface_points = np.c_[surface_points["R"], surface_points["Z"]]
        surface_points = surface_points - axis_points
        surface_angles = np.atan2(surface_points[:,1], surface_points[:,0])
        surface_distances = np.linalg.norm(surface_points, axis=1)

        distance_basis = DoubleFourierSeries(M=20, N=16, NFP=2, sym='cos')
        distance_transform = Transform(
            Grid(
                np.c_[np.ones_like(surface_angles), surface_angles, sample_grid.nodes[:,2]], 
                sort=False, 
                jitable=True
            ), 
            distance_basis, 
            # build=False, 
            # build_pinv=True,
            build=True, 
            build_pinv=False,
            method="direct1"
        )
        # Db_lmn = distance_transform.fit(surface_distances)

        weights = np.ones_like(weights)
        W = np.diag(weights)
        A = distance_transform.matrices[distance_transform.method][0][0][0]
        b = np.log2(surface_distances)
        Db_lmn = np.linalg.lstsq(W @ A, weights * b, rcond=None)[0]

        transformed_distance = Transform(
            sample_grid,
            distance_basis, 
            build=True,
        ).transform(Db_lmn, 0, 0, 0)
        transformed_distance = 2.0 ** transformed_distance

        self.add_input("axis_R_1to4", axis.R_n[1:5], type=DType.vec4)
        self.add_input("axis_R_5to8", axis.R_n[5:9], type=DType.vec4)
        self.add_input("axis_Z_1to4", axis.Z_n[::-1][:4], type=DType.vec4)
        self.add_input("axis_Z_5to8", axis.Z_n[::-1][4:8], type=DType.vec4)

        self.add_input("D_coef_msin_1to4_nsin_1to4", np.array([
            -0.057843789797881774, -0.01760730780195105, -0.0029522548523150116, -0.00016450832691972117, 
            0.7768144362879181, 0.21057176382106962, 0.043157271596269794, 0.007313800999116674, 
            -0.06813209783848478, -0.0804245189151688, -0.026033203426114745, -0.005578173956241869, 
            -0.06113922795078147, -0.1809525304865514, -0.09669581186399909, -0.03331965898027024
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_msin_5to8_nsin_1to4", np.array([
            -0.01641884234435244, 0.06606271285540166, 0.08186483782561708, 0.04338426762157256, 
            -0.009939052964254444, 0.028631470400256667, 0.05568519488513659, 0.036782290228162234, 
            0.013235018669377707, -0.0023014440429341004, -0.0763171838718387, -0.10294722936583253, 
            0.0029969223845392884, 0.0018628856179709469, 0.0009217531231761195, 0.016581535259024338
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_msin_1to4_nsin_5to8", np.array([
            9.223482885178935e-05, 5.933389418889823e-05, 2.2497223393647126e-05, -1.6674252047830032e-05, 
            0.0010439877270293774, 0.00016163872714583527, 4.006923029420406e-05, -2.1519049976270743e-05, 
            -0.0010620973047274496, -0.0002783874765174277, -0.00015581940122287585, -0.00011221809413108907, 
            -0.008974505573046743, -0.001973700265891276, -0.000333293486240005, 0.00010460918361676619
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_msin_5to8_nsin_5to8", np.array([
            0.015712175798224204, 0.0046243455813683165, 0.0014055278848807418, 0.0007647168776222513, 
            0.015920549176904414, 0.005087996692363095, 0.000849917685974734, -0.0007937977019566722, 
            -0.07092005982238558, -0.032731271880434565, -0.011100587594802924, -0.0028798591068766657, 
            0.030089357484686663, 0.026689968697865413, 0.015491385578877803, 0.006255230349526924
        ]).reshape(4,4), type=DType.mat4)

        self.add_input("D_coef_mcos_1to4_ncos_1to4", np.array([
            0.01498787631439083, 0.0173707474850388, 0.004012188382260459, 0.0003905164844689446, 
            -0.6680148582938907, -0.21224901191733603, -0.04436095703920831, -0.007585720539890185, 
            0.07947393306611356, 0.0786183653100811, 0.026107463853367804, 0.00566105338972242, 
            0.0605753098182, 0.18232633771723777, 0.09679075580522684, 0.03336890917963531
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_mcos_5to8_ncos_1to4", np.array([
            0.01181451913373055, -0.06554614947994854, -0.0820678061503728, -0.04339801959721671, 
            0.007277721280315896, -0.028094798522035766, -0.05569374723157257, -0.03680800608469301, 
            -0.012487441224517677, 0.002050539934688962, 0.07636902961727057, 0.10294119249777016, 
            -0.0028471987449018715, -0.0021245114437260515, -0.0008900428575595934, -0.016560672537618476
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_mcos_1to4_ncos_5to8", np.array([
            -7.061723596832326e-05, -5.130382684661976e-05, -1.4585461877075767e-05, 1.2683891031539662e-05, 
            -0.0011048461631383288, -0.00016999483075174487, -2.9600352258171703e-05, 2.8800285549458043e-05, 
            0.0010760085873420255, 0.00027530368345780876, 0.00014994496330647533, 0.00011412395862216845, 
            0.008969058578111935, 0.0019699691423098603, 0.0003342579203347712, -0.00010201616816271197
        ]).reshape(4,4), type=DType.mat4)
        self.add_input("D_coef_mcos_5to8_ncos_5to8", np.array([
            -0.01569972189048538, -0.004624228031592639, -0.0014060135336530187, -0.0007670518208377081, 
            -0.015911920532362095, -0.005085936506545846, -0.0008517053999580054, 0.0007937965287029697, 
            0.0709067644044046, 0.03273313447817568, 0.011102496469700657, 0.002879091092131872, 
            -0.03009139273827692, -0.02669191201754821, -0.015490299649513928, -0.006254997415328756
        ]).reshape(4,4), type=DType.mat4)

        self.add_input("D_coef_m0_ncos_1to4", [-0.2346514075306727, -0.042640112597083446, -0.004619084698109677, -0.00028765522826017936], type=DType.vec4)
        self.add_input("D_coef_m0_ncos_5to8", [4.9297301001165195e-05, 3.61168776752685e-05, -4.1286509573496e-06, -2.525624527660586e-05], type=DType.vec4)
        self.add_input("D_coef_mcos_1to4_n0", [0.0854591279838931, -0.12388343549097836, -0.027036621864115762, -0.028389002144220053], type=DType.vec4)
        self.add_input("D_coef_mcos_5to8_n0", [0.005494500151333792, 0.0033808583222173713, 0.0011653368793052322, 0.000768403574803983], type=DType.vec4)
        self.add_input("D_coef_m0_n0", -2.8170444055548916, type=DType.float)
        self.add_input("R0", [axis.R_n[0], 0.0], type=DType.vec2)

    def sdf_definition(self, p: Op):
        # first need to compute zeta
        zeta = atan(p.y, p.x) * 2.0
        zeta_cos_1to4 = cos(vec4(zeta, 2.0 * zeta, 3.0 * zeta, 4.0 * zeta))
        zeta_cos_5to8 = cos(vec4(5.0 * zeta, 6.0 * zeta, 7.0 * zeta, 8.0 * zeta))
        zeta_sin_1to4 = sin(vec4(zeta, 2.0 * zeta, 3.0 * zeta, 4.0 * zeta))
        zeta_sin_5to8 = sin(vec4(5.0 * zeta, 6.0 * zeta, 7.0 * zeta, 8.0 * zeta))
        r0 = self.R0
        r0 = vec2(
            r0.x + dot(zeta_cos_1to4, self.axis_R_1to4) + dot(zeta_cos_5to8, self.axis_R_5to8),
            r0.y + dot(zeta_sin_1to4, self.axis_Z_1to4) + dot(zeta_sin_5to8, self.axis_Z_5to8)
        )
        r = vec2(length(p.xy), p.z) - r0

        # Good until here

        theta = atan(r.y, r.x)
        theta_cos_1to4 = cos(vec4(theta, 2.0 * theta, 3.0 * theta, 4.0 * theta))
        theta_cos_5to8 = cos(vec4(5.0 * theta, 6.0 * theta, 7.0 * theta, 8.0 * theta))
        theta_sin_1to4 = sin(vec4(theta, 2.0 * theta, 3.0 * theta, 4.0 * theta))
        theta_sin_5to8 = sin(vec4(5.0 * theta, 6.0 * theta, 7.0 * theta, 8.0 * theta))

        D =      dot(self.D_coef_msin_1to4_nsin_1to4 * theta_sin_1to4, zeta_sin_1to4)
        D +=      dot(self.D_coef_msin_5to8_nsin_1to4 * theta_sin_5to8, zeta_sin_1to4)
        D +=      dot(self.D_coef_msin_1to4_nsin_5to8 * theta_sin_1to4, zeta_sin_5to8)
        D +=      dot(self.D_coef_msin_5to8_nsin_5to8 * theta_sin_5to8, zeta_sin_5to8)

        D +=      dot(self.D_coef_mcos_1to4_ncos_1to4 * theta_cos_1to4, zeta_cos_1to4)
        D +=      dot(self.D_coef_mcos_5to8_ncos_1to4 * theta_cos_5to8, zeta_cos_1to4)
        D +=      dot(self.D_coef_mcos_1to4_ncos_5to8 * theta_cos_1to4, zeta_cos_5to8)
        D +=      dot(self.D_coef_mcos_5to8_ncos_5to8 * theta_cos_5to8, zeta_cos_5to8)

        D +=      dot(self.D_coef_mcos_1to4_n0, theta_cos_1to4)
        D +=      dot(self.D_coef_mcos_5to8_n0, theta_cos_5to8)

        D +=      dot(self.D_coef_m0_ncos_1to4, zeta_cos_1to4)
        D +=      dot(self.D_coef_m0_ncos_5to8, zeta_cos_5to8)

        D +=      self.D_coef_m0_n0

        D = exp2(D)

        return length(r) - D


show_part(
    frzsurface(eq.surface),
    [0.07, 0.3, 0.5]
)

# show_part(
#     intersect(
#         # cylinder(0.5, 1.0),
#         translate(cylinder(0.5, 1.0), [0.2, 0, 0]),
#         frzsurface(eq.surface)
#     ),
#     [0.4, 0.02, 0.08]
# )

# p = Op(OpType.CONST, ('p',), DType.vec3, value='p')
# g = frzsurface(eq.surface)(p)
# g = cylinder(0.5, 0.5)(p)

# dot = graphviz.Digraph(comment='SDF parse tree')
# dot.attr(rankdir='LR', dpi='300')
# def _traverse(parent: Op, op: Op, seen: set[tuple[str, str]]):
#     if op.opcode != OpType.CONST:
#         name = f'{op.opcode}'.removeprefix('OpType.')
#         shape = None
#     else:
#         name = f'{op.name if op.name != '' else op.value}'
#         shape = 'box'
#     dot.node(f'{hash(op)}', name, shape=shape)
#     edge = (f'{hash(op)}', f'{hash(parent)}')
#     if parent is not None and edge not in seen:
#         dot.edge(*edge, label=f'{op.rettype.dtype}'.removeprefix('DType.'))
#         seen.add(edge)
#     if op.opcode != OpType.CONST:
#         for a in op.args: _traverse(op, a, seen)

# def traverse(op: Op): return _traverse(None, op, set())

# traverse(g)

# dot.render('sdf_parse_tree', format='png', view=True)

#show_part(
#    cylinder(0.2, 1.0),
#    [0.6, 0.4, 0.2]
#)

# print(Builder(make_part()).build())
# print(eq.surface.R_lmn.shape)
# R_mode_numbers = eq.surface.R_basis.modes[eq.surface.R_basis.modes[:,1]==0][:,2]
# R_mode_arr_idx = np.where(eq.surface.R_basis.modes[:,1]==0)[0]
# print(R_mode_numbers)
# print(R_mode_arr_idx)
