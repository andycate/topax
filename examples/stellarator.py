"""
The goal of this example is to show how a DESC equilibrium can be
directly represented as an SDF!
"""

import numpy as np
from topax.sdfs import SDF
from topax.ops import Const, Op, OpType
from topax._builders import Builder

from desc.geometry import FourierRZToroidalSurface
import desc.examples

eq = desc.examples.get("precise_QA")

class frzsurface(SDF):
    def __init__(self, surface: FourierRZToroidalSurface):
        # send all mode magnitudes as array to shader
        # will need to have a way to check if mode numbers/symmetry changes
        # so that shader can be re-generated when that happens
        # and that will require changing how SDF hashing works
        # the way graphs are generated will need to change to allow
        # array constants, not just vectors and floats
        self.surface = surface
        super().__init__()
        self.add_arrays(r_lmn=surface.R_lmn, z_lmn=surface.Z_lmn)

    def sdf_definition(self, p):
        # first need to compute zeta
        zeta = Op(OpType.ATAN, p.y, p.x) # / float(self.surface.NFP)
        # next we need to compute the (r,z) coordinate of the closest point 
        # on the central axis by computing the fourier series for all 
        # modes where m = 0
        R_mode_numbers = self.surface.R_basis.modes[self.surface.R_basis.modes[:,1]==0][:,2]
        R_mode_arr_idx = np.where(self.surface.R_basis.modes[:,1]==0)[0]
        r = self.r_lmn[int(R_mode_arr_idx[np.where(R_mode_numbers==0)[0].item()])]
        for i, mode in enumerate(R_mode_numbers):
            if mode == 0: continue
            if mode > 0:
                r += 2.0 * self.r_lmn[int(R_mode_arr_idx[i])] * Op(OpType.COS, float(mode) * zeta)
            else:
                r += 2.0 * self.r_lmn[int(R_mode_arr_idx[i])] * Op(OpType.SIN, (-float(mode)) * zeta)

        Z_mode_numbers = self.surface.Z_basis.modes[self.surface.Z_basis.modes[:,1]==0][:,2]
        Z_mode_arr_idx = np.where(self.surface.Z_basis.modes[:,1]==0)[0]
        z = 0.0
        for i, mode in enumerate(Z_mode_numbers):
            if mode == 0: raise ValueError("should be no zeroth mode of Z")
            if mode > 0:
                z += 2.0 * self.z_lmn[int(Z_mode_arr_idx[i])] * Op(OpType.COS, float(mode) * zeta)
            else:
                z += 2.0 * self.z_lmn[int(Z_mode_arr_idx[i])] * Op(OpType.SIN, (-float(mode)) * zeta)

        return Op(OpType.LEN, Op(OpType.MAKE_VEC2, r - Op(OpType.LEN, p.xy), z - p.z)) - 0.25
        # next we will find p_prime, which is (R,Z), where R is relative
        # to the closest point on the central axis
        pass
        # finally we will do something clever to compute the point on the
        # surface that has the same normal as the point p_prime, and then
        # subtract the vectors and take the length to get final distance
        pass

def make_part():
    return frzsurface(eq.surface)

# print(Builder(make_part()).build())
# print(eq.surface.R_lmn.shape)
# R_mode_numbers = eq.surface.R_basis.modes[eq.surface.R_basis.modes[:,1]==0][:,2]
# R_mode_arr_idx = np.where(eq.surface.R_basis.modes[:,1]==0)[0]
# print(R_mode_numbers)
# print(R_mode_arr_idx)

