import abc

from numpy.typing import ArrayLike

from topax._utils import resolve_type
from topax.ops import Op, OpType, Const

class SDF(abc.ABC):
    def __init__(self, *args, **kwargs):
        assert len(args) == 0
        self._values = {}
        for k in kwargs:
            item = kwargs[k]
            self._values[k] = item
            v = Const(self, k, resolve_type(item))
            setattr(self, k, v)

    @abc.abstractmethod
    def sdf_definition(self, p) -> Op | Const:
        raise NotImplementedError()

    def __call__(self, p):
        return self.sdf_definition(p)
    
    def __getitem__(self, key):
        return self._values[key]

class empty(SDF):
    def __init__(self):
        super().__init__()

    def sdf_definition(self, p):
        return Const(None, 'uintBitsToFloat(0x7F800000u)', 'float')

class sphere(SDF):
    def __init__(
        self,
        radius: float, 
        center: ArrayLike | None=None, 
        x: float | None=None,
        y: float | None=None,
        z: float | None=None
    ):
        if center is not None:
            super().__init__(radius=radius, center=center)
        else:
            center = [x, y, z]
            if all(e is None for e in center):
                super().__init__(radius=radius)
            else:
                center = [0.0 if e is None else e for e in center]
                super().__init__(radius=radius, center=center)

    def sdf_definition(self, p):
        if hasattr(self, 'center'):
            return Op(OpType.LEN, p - self.center) - self.radius
        else:
            return Op(OpType.LEN, p) - self.radius
        
class box(SDF):
    def __init__(
        self,
        size: ArrayLike,
    ):
        super().__init__(size=size)

    def sdf_definition(self, p):
        q = Op(OpType.ABS, p) - self.size
        return Op(OpType.LEN, Op(OpType.MAX, q, 0.0)) + Op(OpType.MIN, Op(OpType.MAX, q.x, Op(OpType.MAX, q.y, q.z)), 0.0)
    
class translate(SDF):
    def __init__(self, sdf: SDF, offset: ArrayLike):
        self.sdf = sdf
        super().__init__(offset=offset)

    def sdf_definition(self, p):
        return self.sdf(p - self.offset)
    
class union(SDF):
    def __init__(self, *sdfs: SDF):
        self.sdfs = sdfs
        super().__init__()

    def sdf_definition(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        oper = Op(OpType.MIN, self.sdfs[0](p), self.sdfs[1](p))
        for i in range(2, len(self.sdfs)):
            oper = Op(OpType.MIN, oper, self.sdfs[i](p))
        return oper
    
class intersect(SDF):
    def __init__(self, *sdfs: SDF):
        self.sdfs = sdfs
        super().__init__()

    def sdf_definition(self, p):
        if len(self.sdfs) == 1:
            return self.sdfs[0](p)
        oper = Op(OpType.MAX, self.sdfs[0](p), self.sdfs[1](p))
        for i in range(2, len(self.sdfs)):
            oper = Op(OpType.MAX, oper, self.sdfs[i](p))
        return oper
    
class subtract(SDF):
    def __init__(self, sdf: SDF, tool: SDF):
        self.sdf = sdf
        self.tool = tool
        super().__init__()

    def sdf_definition(self, p):
        return Op(OpType.MAX, self.sdf(p), -self.tool(p))
    
class scale(SDF):
    def __init__(self, sdf: SDF, amount: float):
        self.sdf = sdf
        super().__init__(amount=amount)

    def sdf_definition(self, p):
        return self.sdf(p / self.amount) * self.amount
