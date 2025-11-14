"""
The goal of this example is to show how a DESC equilibrium can be
directly represented as an SDF!
"""

import numpy as np
import graphviz
from topax import show_part
from topax.sdfs import SDF, offset, gyroid, intersect, cylinder, scale, box
from topax.ops import OpBase, sin, cos, vec2, vec4, atan, dot, length, exp2
from topax.types import DType, BaseType

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.basis import DoubleFourierSeries
from desc.transform import Transform
from desc.grid import LinearGrid, Grid
import desc.examples

eq = desc.examples.get("precise_QA")

class frzsurface(SDF):
    def __init__(self, eq: Equilibrium, surface: FourierRZToroidalSurface=None):
        # self.surface = surface

        axis = eq.get_axis()
        if not surface: surface = eq.surface
        assert axis.N == 8
        assert eq.N == 8
        assert eq.M == 8
        assert eq.NFP == 2

        N=eq.N*2+1
        M=eq.M*16+1

        sample_grid = LinearGrid(rho=[1.0], N=N, M=M, NFP=2)
        axis_points = axis.compute(["R", "Z"], grid=sample_grid)
        axis_points = np.c_[axis_points["R"], axis_points["Z"]]
        surface_points = surface.compute(["R", "Z"], grid=sample_grid)
        weights = surface.compute(["|e_theta x e_zeta|"], grid=sample_grid)["|e_theta x e_zeta|"]
        surface_points = np.c_[surface_points["R"], surface_points["Z"]]
        surface_points = surface_points - axis_points
        surface_angles = np.atan2(surface_points[:,1], surface_points[:,0])
        surface_distances = np.linalg.norm(surface_points, axis=1)

        distance_basis = DoubleFourierSeries(M=8, N=8, NFP=2, sym='cos')
        distance_transform = Transform(
            Grid(
                np.c_[np.ones_like(surface_angles), surface_angles, sample_grid.nodes[:,2]], 
                sort=False, 
                jitable=True
            ), 
            distance_basis, 
            build=False, 
            build_pinv=True,
            method="direct1"
        )

        b = np.log2(surface_distances)
        distance_transform.fit(b)
        Db_lmn = distance_transform.fit(b)

        m0_coef = Db_lmn[np.where(np.all([distance_transform.modes[:,1] == 0, distance_transform.modes[:,2] != 0], axis=0))[0]]
        n0_coef = Db_lmn[np.where(np.all([distance_transform.modes[:,2] == 0, distance_transform.modes[:,1] != 0], axis=0))[0]]
        n0_m0_coef = Db_lmn[np.where(np.all([distance_transform.modes[:,2] == 0, distance_transform.modes[:,1] == 0], axis=0))[0]].item()

        g = np.mgrid[-8:9,-8:9].astype(np.int32).T.reshape(-1,2)
        mn_coef = np.zeros((16,16))
        for p in g:
            if p[0] == 0 or p[1] == 0: continue
            if p[0] < 0: ixm = -p[0]-1
            else: ixm = p[0] + 7
            if p[1] < 0: ixn = -p[1]-1
            else: ixn = p[1] + 7
            i = np.where(np.all([p[0] == distance_transform.modes[:,1], p[1] == distance_transform.modes[:,2]], axis=0))[0]
            if len(i) == 0: continue
            else: i = i.item()
            mn_coef[ixm,ixn] = Db_lmn[i]

        self.axis_R_1to4 = self.add_param(axis.R_n[1:5], dtype=DType(BaseType.vec4))
        self.axis_R_5to8 = self.add_param(axis.R_n[5:9], dtype=DType(BaseType.vec4))
        self.axis_Z_1to4 = self.add_param(axis.Z_n[::-1][:4], dtype=DType(BaseType.vec4))
        self.axis_Z_5to8 = self.add_param(axis.Z_n[::-1][4:8], dtype=DType(BaseType.vec4))

        self.D_coef_msin_1to4_nsin_1to4 = self.add_param(mn_coef[:4,:4], dtype=DType(BaseType.mat4))
        self.D_coef_msin_5to8_nsin_1to4 = self.add_param(mn_coef[4:8,:4], dtype=DType(BaseType.mat4))
        self.D_coef_msin_1to4_nsin_5to8 = self.add_param(mn_coef[:4,4:8], dtype=DType(BaseType.mat4))
        self.D_coef_msin_5to8_nsin_5to8 = self.add_param(mn_coef[4:8,4:8], dtype=DType(BaseType.mat4))

        self.D_coef_mcos_1to4_ncos_1to4 = self.add_param(mn_coef[8:12,8:12], dtype=DType(BaseType.mat4))
        self.D_coef_mcos_5to8_ncos_1to4 = self.add_param(mn_coef[12:16,8:12], dtype=DType(BaseType.mat4))
        self.D_coef_mcos_1to4_ncos_5to8 = self.add_param(mn_coef[8:12,12:16], dtype=DType(BaseType.mat4))
        self.D_coef_mcos_5to8_ncos_5to8 = self.add_param(mn_coef[12:16,12:16], dtype=DType(BaseType.mat4))

        self.D_coef_m0_ncos_1to4 = self.add_param(m0_coef[:4], dtype=DType(BaseType.vec4))
        self.D_coef_m0_ncos_5to8 = self.add_param(m0_coef[4:8], dtype=DType(BaseType.vec4))
        self.D_coef_mcos_1to4_n0 = self.add_param(n0_coef[:4], dtype=DType(BaseType.vec4))
        self.D_coef_mcos_5to8_n0 = self.add_param(n0_coef[4:8], dtype=DType(BaseType.vec4))
        self.D_coef_m0_n0 = self.add_param(n0_m0_coef, dtype=DType(BaseType.float))
        self.R0 = self.add_param([axis.R_n[0], 0.0], dtype=DType(BaseType.vec2))

        super().__init__()

    def opdef(self, p: OpBase):
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
    # intersect(offset(frzsurface(eq), 0.1), scale(gyroid(), 0.1)),
    frzsurface(eq, eq.surface.constant_offset_surface(0.2, grid=LinearGrid(N=eq.N*4+1, M=eq.M*4+1))).i(gyroid().s(0.1)).u(cylinder(0.5, 1)).u(box(x=2.2, y=0.2, z=0.2)),
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
