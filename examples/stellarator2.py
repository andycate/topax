import numpy as np
from topax import show_part
from topax.sdfs_old import SDF, intersect, box, translate, cylinder
import topax.ops_old as ops_old

from desc.geometry import FourierRZToroidalSurface
import desc.examples

eq = desc.examples.get("DSHAPE")

class frzsurface(SDF):
    def __init__(self, surface: FourierRZToroidalSurface, r0: float):
        self.surface = surface
        self.add_input('r_lmn', surface.R_lmn, ops_old.RetType(ops_old.DType.float, len(surface.R_lmn)))
        self.add_input('z_lmn', surface.Z_lmn, ops_old.RetType(ops_old.DType.float, len(surface.Z_lmn)))
        self.add_input('r0', r0, ops_old.DType.float)

    def sdf_definition(self, p) -> ops_old.Op:
        zeta = ops_old.atan(p.y, p.x) * float(self.surface.NFP)
        r0_idx = int(np.where(np.all(self.surface.R_basis.modes[:,1:] == 0, axis=1))[0][0])
        # r0 = self.r_lmn[r0_idx]
        r0 = self.r0
        r_prime = ops_old.vec2(ops_old.length(p.xy) - r0, p.z)
        theta = ops_old.atan(r_prime.y, r_prime.x) #TODO: check order

        R = 0.0
        for i, mode in enumerate(reversed(self.surface.R_basis.modes)):
            i = len(self.surface.R_basis.modes) - i - 1
            print(i)
            m = mode[1]
            n = mode[2]
            if n == 0 and m == 0: continue
            coef = self.r_lmn[i]
            # if abs(m) > 0 and abs(n) > 0: continue
            if m >= 0 and n >= 0:
                R += coef * ops_old.cos(abs(float(m)) * theta) * ops_old.cos(abs(float(n)) * zeta)
            elif m >= 0 and n < 0:
                R += coef * ops_old.cos(abs(float(m)) * theta) * ops_old.sin(abs(float(n)) * zeta)
            elif m < 0 and n >= 0:
                R += coef * ops_old.sin(abs(float(m)) * theta) * ops_old.cos(abs(float(n)) * zeta)
            elif m < 0 and n < 0:
                R += coef * ops_old.sin(abs(float(m)) * theta) * ops_old.sin(abs(float(n)) * zeta)
            else: raise ValueError(f"mode numbers {m},{n} not recognized")
        
        Z = 0.0
        for i, mode in enumerate(reversed(self.surface.Z_basis.modes)):
            i = len(self.surface.Z_basis.modes) - i - 1
            m = mode[1]
            n = mode[2]
            # if abs(m) > 0 and abs(n) > 0: continue
            coef = self.z_lmn[i]
            if m >= 0 and n >= 0:
                Z += coef * ops_old.cos(abs(float(m)) * theta) * ops_old.cos(abs(float(n)) * zeta)
            elif m >= 0 and n < 0:
                Z += coef * ops_old.cos(abs(float(m)) * theta) * ops_old.sin(abs(float(n)) * zeta)
            elif m < 0 and n >= 0:
                Z += coef * ops_old.sin(abs(float(m)) * theta) * ops_old.cos(abs(float(n)) * zeta)
            elif m < 0 and n < 0:
                Z += coef * ops_old.sin(abs(float(m)) * theta) * ops_old.sin(abs(float(n)) * zeta)
            else: raise ValueError(f"mode numbers {m},{n} not recognized")

        return ops_old.length(r_prime) - ops_old.length(ops_old.vec2(R, Z))


show_part(
    intersect(
        frzsurface(eq.surface, 3.11),
        translate(
            box([5.0, 4.0, 2.0]),
            [0.0, 4.0, 0.0]
        )
    ),
    [0.4, 0.5, 0.7]
)

r0_idx = int(np.where(np.all(eq.surface.R_basis.modes[:,1:] == 0, axis=1))[0][0])
r0 = eq.surface.R_lmn[r0_idx]

show_part(
    translate(cylinder(r=r0, h=1.0), [0.0, 1.1, 0.0]),
    [0.4, 0.7, 0.2]
)
