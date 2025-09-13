from cax.sdfs import Op, OpTypes, Const, SDF, sphere, translate, union


# p = Op(OpTypes.CONST, "p")

# optree = part(p)

# print(optree)

class Builder:
    def __init__(self):
        self.input_vars = {}
        self.local_vars = {"p": "vec3"}
        self.unused_local_vars = {}
        self.function_body = []

    def get_type_const(self, val):
        if isinstance(val, float):
            return 'float'
        if isinstance(val, tuple):
            match len(val):
                case 3:
                    return 'vec3'
                case 2:
                    return 'vec2'
                case _:
                    raise NotImplementedError(f"list of length {len(val)} not supported")
        raise NotImplementedError(f"const of type {type(val)} not supported")
    
    def get_local_var(self, type, *used_vars):
        for v in used_vars:
            if v.startswith("sdfin") or v == 'p':
                continue
            if self.local_vars[v] == type:
                print("returning ", v)
                return v
        for v in self.unused_local_vars:
            if self.unused_local_vars[v] == type:
                del self.unused_local_vars[v]
                return v
        new_var_name = f"local_var{len(self.local_vars.keys())}"
        self.local_vars[new_var_name] = type
        self.unused_local_vars[new_var_name] = type
        print("after adding ", self.local_vars)
        return new_var_name
    
    def release_local_var(self, name):
        print("before removing ", self.local_vars)
        if name.startswith("sdfin"): return
        self.unused_local_vars[name] = self.local_vars[name]

    def build(self, optree: Op | Const):
        if isinstance(optree, Const):
            if optree not in self.input_vars:
                if isinstance(optree.lhs, str):
                    self.input_vars[optree] = (optree.lhs, None)
                else:
                    self.input_vars[optree] = (f"sdfin_var{len(self.input_vars.keys())}", self.get_type_const(optree.lhs))
            return self.input_vars[optree][0]
        else:
            assert isinstance(optree.lhs, Op)
            lhs_varname = self.build(optree.lhs)
            rhs_varname = None
            if optree.rhs is not None:
                assert isinstance(optree.rhs, Op)
                rhs_varname = self.build(optree.rhs)
            match optree.opcode:
                case OpTypes.ADD:
                    assert rhs_varname is not None
                    if self.local_vars[lhs_varname] == 'vec3' or self.local_vars[rhs_varname] == 'vec3':
                        out_var = self.get_local_var('vec3', lhs_varname, rhs_varname)
                    elif self.local_vars[lhs_varname] == 'vec2' or self.local_vars[rhs_varname] == 'vec2':
                        out_var = self.get_local_var('vec2', lhs_varname, rhs_varname)
                    else:
                        out_var = self.get_local_var('float', lhs_varname, rhs_varname)
                    self.function_body.append(f"{out_var} = {lhs_varname} + {rhs_varname};")
                    if lhs_varname != out_var:
                        self.release_local_var(lhs_varname)
                    if rhs_varname != out_var:
                        self.release_local_var(rhs_varname)
                    return out_var
                case OpTypes.SUB:
                    assert rhs_varname is not None
                    if self.local_vars[lhs_varname] == 'vec3' or self.local_vars[rhs_varname] == 'vec3':
                        out_var = self.get_local_var('vec3', lhs_varname, rhs_varname)
                    elif self.local_vars[lhs_varname] == 'vec2' or self.local_vars[rhs_varname] == 'vec2':
                        out_var = self.get_local_var('vec2', lhs_varname, rhs_varname)
                    else:
                        out_var = self.get_local_var('float', lhs_varname, rhs_varname)
                    self.function_body.append(f"{out_var} = {lhs_varname} - {rhs_varname};")
                    if lhs_varname != out_var:
                        self.release_local_var(lhs_varname)
                    if rhs_varname != out_var:
                        self.release_local_var(rhs_varname)
                    print("returning from sub ", out_var)
                    return out_var
                case OpTypes.LEN:
                    assert rhs_varname is None
                    if self.local_vars[lhs_varname] == 'vec3':
                        out_var = self.get_local_var('vec3', lhs_varname)
                    else:
                        out_var = self.get_local_var('vec2', lhs_varname)
                    self.function_body.append(f"{out_var} = length({lhs_varname});")
                    if lhs_varname != out_var:
                        self.release_local_var(lhs_varname)
                    print("returning from len ", out_var)
                    return out_var
                case OpTypes.MIN:
                    assert rhs_varname is not None
                    if self.local_vars[lhs_varname] == 'vec3' or self.local_vars[rhs_varname] == 'vec3':
                        out_var = self.get_local_var('vec3', lhs_varname, rhs_varname)
                    elif self.local_vars[lhs_varname] == 'vec2' or self.local_vars[rhs_varname] == 'vec2':
                        out_var = self.get_local_var('vec2', lhs_varname, rhs_varname)
                    else:
                        out_var = self.get_local_var('float', lhs_varname, rhs_varname)
                    self.function_body.append(f"{out_var} = min({lhs_varname}, {rhs_varname});")
                    if lhs_varname != out_var:
                        self.release_local_var(lhs_varname)
                    if rhs_varname != out_var:
                        self.release_local_var(rhs_varname)
                    print("returning from min ", out_var)
                    return out_var
                case _:
                    raise NotImplementedError(f"Opcode {optree.opcode} has not been implemented")
                

    def make_shader(self, sdf: SDF):
        p = Op(OpTypes.CONST, "p")
        part = sdf(p)

        out_var = self.build(part)
        self.function_body.append(f"return {out_var};")

        for v in reversed(self.local_vars):
            self.function_body.insert(0, f"{self.local_vars[v]} {v};")


        print("\n".join([f"uniform {self.input_vars[v][1]} {self.input_vars[v][0]};" for v in self.input_vars if self.input_vars[v][1] is not None]))
        print("------------")
        print("\n".join(self.function_body))


part = union(
    sphere(0.5, (0.3, 0, 0)),
    sphere(0.5, (-0.3, 0, 0)),
)
p = Const(None, 'p', 'vec3')
part1_hash = part(p)

part = union(
    sphere(0.6, (0.3, 0.1, 0)),
    sphere(0.2, (-0.3, 0.3, 0)),
)
p = Const(None, 'p', 'vec3')
part2_hash = part(p)

print(repr(part1_hash) == repr(part2_hash))
