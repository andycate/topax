import jinja2

from topax._utils import make_shader_const
from topax.ops import Const, Op, OpType

class Builder:
    template = jinja2.Environment(loader=jinja2.PackageLoader('topax')).get_template('shader.glsl.j2')
    def __init__(self, sdf):
        self.sdf = sdf
        self.p = Const(None, 'p', 'vec3')
        self.optree = self.sdf(self.p)
        self.input_vars = {} # keyed by Const objects, value is varname
        self.input_vars_types = {} # keyed by input var name, value is type
        self.local_vars = {} # keyed by name of var, element is type
        self.local_vars_unused = {} # keyed by type, each element is list
        self.lines = []

    def _allocate_var(self, vartype):
        if vartype not in self.local_vars_unused:
            self.local_vars_unused[vartype] = []
        if len(self.local_vars_unused[vartype]) > 0:
            return self.local_vars_unused[vartype].pop()
        else:
            new_varname = f"local_var{len(self.local_vars.keys())}"
            self.local_vars[new_varname] = vartype
            self.lines.append(f"{vartype} {new_varname};")
            return new_varname

    def _release_var(self, varname):
        if varname is not None:
            if varname in self.input_vars_types: return
            assert varname in self.local_vars
            vartype = self.local_vars[varname]
            self.local_vars_unused[vartype].append(varname)

    def _parse_input_vars(self, optree: Op | Const):
        if isinstance(optree, Const):
            if optree in self.input_vars: return
            if optree.sdf is None:
                # self.input_vars[optree] = optree.param
                # self.input_vars_types[optree.param] = optree.rettype
                pass
            else:
                new_name = f"sdfin_var{len(self.input_vars.keys())}"
                self.input_vars[optree] = new_name
                self.input_vars_types[new_name] = optree.rettype
        else:
            self._parse_input_vars(optree.lhs)
            if optree.rhs is not None: self._parse_input_vars(optree.rhs)

    def parse_input_vars(self):
        self._parse_input_vars(self.optree)
    
    def get_input_vars(self):
        return {value: key for key, value in self.input_vars.items()}

    def _parse_ops(self, optree: Op | Const, out_varname: str):
        # deal with case for empty or constant SDF
        if isinstance(optree, Const):
            assert optree.sdf is None
            self.lines.append(f"{out_varname} = {optree.param};")
            return
        
        # retrieve shader var name for lhs
        lhs_is_external_param = False
        if isinstance(optree.lhs, Const):
            if optree.lhs.sdf is None:
                lhs_is_external_param = True
                if isinstance(optree.lhs.param, str): lhs_varname = optree.lhs.param
                else: lhs_varname = make_shader_const(optree.lhs.param, optree.lhs.rettype)
            else:
                lhs_varname = self.input_vars[optree.lhs]
        else:
            lhs_varname = self._allocate_var(optree.lhs.rettype)
            self._parse_ops(optree.lhs, lhs_varname)
        
        # retrieve shader var name for rhs
        rhs_is_external_param = False
        if optree.rhs is not None:
            if isinstance(optree.rhs, Const):
                if optree.rhs.sdf is None:
                    rhs_is_external_param = True
                    if isinstance(optree.rhs.param, str): rhs_varname = optree.rhs.param
                    else: rhs_varname = make_shader_const(optree.rhs.param, optree.rhs.rettype)
                else:
                    rhs_varname = self.input_vars[optree.rhs]
            else:
                rhs_varname = self._allocate_var(optree.rhs.rettype)
                self._parse_ops(optree.rhs, rhs_varname)
        else:
            rhs_varname = None

        match optree.opcode:
            case OpType.ADD: self.lines.append(f"{out_varname} = {lhs_varname} + {rhs_varname};")
            case OpType.SUB: self.lines.append(f"{out_varname} = {lhs_varname} - {rhs_varname};")
            case OpType.MUL: self.lines.append(f"{out_varname} = {lhs_varname} * {rhs_varname};")
            case OpType.DIV: self.lines.append(f"{out_varname} = {lhs_varname} / {rhs_varname};")
            case OpType.LEN: self.lines.append(f"{out_varname} = length({lhs_varname});")
            case OpType.NORM: self.lines.append(f"{out_varname} = normal({lhs_varname});")
            case OpType.MIN: self.lines.append(f"{out_varname} = min({lhs_varname}, {rhs_varname});")
            case OpType.MAX: self.lines.append(f"{out_varname} = max({lhs_varname}, {rhs_varname});")
            case OpType.NEG: self.lines.append(f"{out_varname} = -{lhs_varname};")
            case OpType.ABS: self.lines.append(f"{out_varname} = abs({lhs_varname});")
            case OpType.X: self.lines.append(f"{out_varname} = {lhs_varname}.x;")
            case OpType.Y: self.lines.append(f"{out_varname} = {lhs_varname}.y;")
            case OpType.Z: self.lines.append(f"{out_varname} = {lhs_varname}.z;")
            case OpType.XY: self.lines.append(f"{out_varname} = {lhs_varname}.xy;")
            case OpType.XZ: self.lines.append(f"{out_varname} = {lhs_varname}.xz;")
            case OpType.YZ: self.lines.append(f"{out_varname} = {lhs_varname}.yz;")
            case _: raise NotImplementedError(f"parsing for opcode {optree.opcode} not supported")

        if not lhs_is_external_param: self._release_var(lhs_varname)
        if not rhs_is_external_param: self._release_var(rhs_varname)

    def build(self):
        self.parse_input_vars()
        self._parse_ops(self.optree, 'd')
        
        return Builder.template.render(
            global_inputs=[dict(name=v, type=self.input_vars_types[v]) for v in self.input_vars_types if v != 'p'],
            lines=self.lines
        )
