"""
The goal of this example is to show how a DESC equilibrium can be
directly represented as an SDF!
"""

from topax.sdfs import SDF
from topax.ops import Const, Op, OpType
from desc.geometry import FourierRZToroidalSurface

class frzsurface(SDF):
    def __init__(self, surface: FourierRZToroidalSurface):
        # send all mode magnitudes as array to shader
        # will need to have a way to check if mode numbers/symmetry changes
        # so that shader can be re-generated when that happens
        # and that will require changing how SDF hashing works
        # the way graphs are generated will need to change to allow
        # array constants, not just vectors and floats
        super().__init__()

    def sdf_definition(self, p):
        # first need to compute zeta
        zeta = Op(OpType.ATAN, p.y, p.x)
        # next we need to compute the r coordinate of the closest point 
        # on the central axis by computing the fourier series for all 
        # modes where m = 0
        pass
        # next we will find p_prime, which is (R,Z), where R is relative
        # to the closest point on the central axis
        pass
        # finally we will do something clever to compute the point on the
        # surface that has the same normal as the point p_prime, and then
        # subtract the vectors and take the length to get final distance
        pass
