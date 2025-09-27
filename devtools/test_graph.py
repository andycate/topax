import topax.ops as ops
from topax.ops import Op, DType, OpType, RetType
from topax.sdfs import SDF, union, sphere, intersect
from pprint import pprint
from collections import deque

class gyroid(SDF):
    def __init__(self, scale: float, fill: float):
        self.add_input('scale', scale, DType.float)
        self.add_input('fill', fill, DType.float)

    def sdf_definition(self, p):
        scaled_p = p * self.scale
        gyroid = ops.abs(ops.dot(ops.sin(scaled_p), ops.cos(scaled_p.yzx))) * 0.33 - self.fill
        return gyroid

# class test_sdf(SDF):
#     def __init__(self, test: float):
#         self.add_input('test', test, DType.float)

#     def sdf_definition(self, p):
#         return p - self.test * 0.33
    

part = union(
    sphere(0.5, [0.2, 0.0, 0.0]), 
    sphere(0.5, [-0.2, 0.0, 0.0])
)

s1 = sphere(0.4, [0.1, 0.0, 0.0])
part2 = union(
    s1, 
    s1,
    sphere(0.5, [-0.2, 0.0, 0.0])
)

part3 = intersect(
    gyroid(1.0, 0.05),
    sphere(5.0)
)
# part3 = gyroid(1.0, 0.05)


# print(part.hash() == part2.hash())

p = Op(OpType.CONST, ('p',), DType.vec3, value='p')
g1 = part(p)
g2 = part2(p)
g3 = part3(p)

# print(repr(g1))
# print(repr(g1))
# print(hash(g2.args[0]) == hash(g2.args[1]))


def traverse(op: Op) -> tuple[dict[Op, int], dict[Op, set[Op]], list[Op]]:
    """
    Traverse an op graph and build reversed graph and node input count, 
    as well as queue ready for tape generation.

    :param op: The operation graph to traverse

    :return in_count: map where key is a sub op and value is number of unique arguments
    :return consumer_nodes: map where key is sub op and value is all ops that consume this op
    :return consts: a list of constants (both static and global vars)
    """
    in_count = {}
    consumer_nodes = {}
    consts = []
    consts_seen = set()

    def _traverse(in_count: dict, consumer_nodes: dict, consts: list, consts_seen: set, op: Op):
        in_cnt = len(set(op.args))
        if op.opcode == OpType.CONST:
            if op not in consts_seen:
                consts.append(op)
                consts_seen.add(op)
        else:
            in_count[op] = in_cnt
            for o in op.args:
                if o not in consumer_nodes: consumer_nodes[o] = set()
                consumer_nodes[o].add(op)
                _traverse(in_count, consumer_nodes, consts, consts_seen, o)

    _traverse(in_count, consumer_nodes, consts, consts_seen, op)
    return in_count, consumer_nodes, consts

def make_tape(in_count: dict[Op, int], consumer_nodes: dict[Op, set[Op]], consts: list[Op]) -> list[Op]:
    """
    Perform a topological sort of the computation graph, generating
    a linear tape
    """
    queue = deque(consts)
    tape = []
    while len(queue) > 0:
        n = queue.popleft()
        if n.opcode != OpType.CONST:
            tape.append(n)
        else:
            assert n in consumer_nodes
            assert n in consts
        if n in consumer_nodes:
            for pn in consumer_nodes[n]:
                in_count[pn] -= 1
                if in_count[pn] == 0:
                    queue.append(pn)
        else:
            assert len(queue) == 0
    return tape

def get_local_vars_ttl(tape: list[Op]) -> dict[Op, int]:
    vars_ttl = {}
    for i, l in enumerate(tape):
        vars_ttl[l] = i
        for arg in l.args:
            if arg.opcode == OpType.CONST: continue
            vars_ttl[arg] = i
    return vars_ttl

def get_static_expression(const: Op) -> str:
    if isinstance(const.value, str): return const.value
    else:
        assert const.rettype.length is None
        match const.rettype.dtype:
            case DType.float: return f"{float(const.value)}"
            case DType.vec2: return f"vec2({float(const.value[0])},{float(const.value[1])})"
            case DType.vec3: return f"vec3({float(const.value[0])},{float(const.value[1])},{float(const.value[2])})"
            case DType.vec4: return f"vec4({float(const.value[0])},{float(const.value[1])},{float(const.value[2])},{float(const.value[3])})"
            case _: raise TypeError(f"expression for type {const.rettype.dtype} not yet supported")

def get_var_definition(name: str, type: RetType) -> str:
    assert type.length is None
    match type.dtype:
        case DType.float: return f"float {name}"
        case DType.vec2: return f"vec2 {name}"
        case DType.vec3: return f"vec3 {name}"
        case DType.vec4: return f"vec4 {name}"
        case _: raise TypeError(f"var definition not supported for type {type}")


def generate_shader_code(graph: Op, prefix: str) -> tuple[list[str], dict[Op, str]]:
    in_count, consumer_nodes, consts = traverse(graph)
    # print(consts)
    tape = make_tape(in_count, consumer_nodes, consts)
    # pprint(tape)
    # for i in range(1, len(tape)):
    #     t = tape[i]
    #     for j in range(i):
    #         t2 = tape[j]
    #         assert t not in t2.args, f"Op {t.opcode} at position {i} required by earlier op {t2.opcode} at position {j}"

    vars_ttl = get_local_vars_ttl(tape)

    lines = [] # the list of strings making up the shader code
    local_expressions = {} # shader string expressions representing ops, key is op value is string (can be local variable name or compute expression)
    local_var_ops = set() # will contain all ops that are stored to a local var
    local_vars_available = {} # local vars available for reuse, key is RetType value is string
    local_vars_total = 0
    global_vars = {c: f"{prefix}_sdfglobal_var{i}" for i, c in enumerate(consts) if c.sdf is not None}

    for ti, op in enumerate(tape):
        assert op not in local_expressions, "duplicate op expression found!"
        arg_expressions = [] # each element is a string for the corresponding op arg, either a variable name or a direct expression
        for arg in op.args:
            # if this is a constant static expression, just convert directly to expression
            if arg.opcode == OpType.CONST and arg.sdf is None: arg_expressions.append(get_static_expression(arg))
            elif arg in local_expressions: # true if this arg is not a global var
                arg_expressions.append(local_expressions[arg])
                if vars_ttl[arg] == ti: # if the last reference of this var is at this line
                    if arg in local_var_ops:
                        local_vars_available[arg.rettype].append(local_expressions[arg])
                        local_var_ops.remove(arg)
                    del local_expressions[arg]
            else:
                try:
                    arg_expressions.append(global_vars[arg])
                except KeyError as e:
                    # print(local_expressions)
                    raise e

        match op.opcode:
            case OpType.ADD: expression = f"({" + ".join(arg_expressions)})"
            case OpType.SUB: expression = f"({arg_expressions[0]} - {arg_expressions[1]})"
            case OpType.MUL: expression = f"({" * ".join(arg_expressions)})"
            case OpType.DIV: expression = f"({arg_expressions[0]} / {arg_expressions[1]})"
            case OpType.LEN: expression = f"length({arg_expressions[0]})"
            case OpType.DOT: expression = f"dot({arg_expressions[0]}, {arg_expressions[1]})"
            case OpType.YZX: expression = f"{arg_expressions[0]}.yzx"
            case OpType.SIN: expression = f"sin({arg_expressions[0]})"
            case OpType.COS: expression = f"cos({arg_expressions[0]})"
            case OpType.ABS: expression = f"abs({arg_expressions[0]})"
            case OpType.MIN:
                # TODO: change this to do binary tree min
                # want to build min compute tree
                # for number of args remaining, find largest power of 2 that is less than this number
                expression = f"min({arg_expressions[0]}, {arg_expressions[1]})"
                for i in range(2, len(arg_expressions)):
                    expression = f"min({expression}, {arg_expressions[i]})"
            case OpType.MAX:
                # TODO: change this to do binary tree min
                # want to build min compute tree
                # for number of args remaining, find largest power of 2 that is less than this number
                expression = f"max({arg_expressions[0]}, {arg_expressions[1]})"
                for i in range(2, len(arg_expressions)):
                    expression = f"max({expression}, {arg_expressions[i]})"
            case _: raise TypeError(f"operation expression for opcode {op.opcode} not yet supported")

        # need to allocate a var if this op gets reused
        if op in consumer_nodes and len(consumer_nodes[op]) > 1:
            if op.rettype not in local_vars_available: local_vars_available[op.rettype] = deque()
            if len(local_vars_available[op.rettype]) > 0:
                out_var_name = local_vars_available[op.rettype].popleft()
                line = out_var_name
            else:
                out_var_name = f"local_var{local_vars_total}"
                local_vars_total += 1
                line = get_var_definition(out_var_name, op.rettype)
            lines.append(f"{line} = {expression};")
            local_expressions[op] = out_var_name
            local_var_ops.add(op)
        else:
            local_expressions[op] = expression

        if ti == len(tape)-1:
            # last expression, return it!
            lines.append(f"return {expression};")
    
    return lines, global_vars

lines, global_vars = generate_shader_code(g3, "sdf1")
# pprint(global_vars)
for k, v in global_vars.items():
    print(v, k)
print()
# print("\n".join(lines))
