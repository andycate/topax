from topax.sdfs import SDF, sphere, intersect, box
from topax.ops import Op, OpType
from topax._builders import Builder

class gyroid(SDF):
    def __init__(self, scale: float, fill: float):
        super().__init__(scale=scale, fill=fill)

    def sdf_definition(self, p):
        scaled_p = p * self.scale
        gyroid = Op(OpType.ABS, Op(OpType.DOT, Op(OpType.SIN, scaled_p), Op(OpType.COS, scaled_p.yzx))) * 0.5 - self.fill
        return gyroid

def make_part():
    return intersect(
        gyroid(2.0, 0.1),
        box([2.0, 2.0, 2.0])
    )
