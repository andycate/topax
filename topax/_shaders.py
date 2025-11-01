from collections import deque
from ordered_set import OrderedSet
from numpy.typing import ArrayLike
import topax.ops as ops
from topax.sdfs import SDF
from topax.types import DType, TypeEnum

class ShaderSDF:
    """Class to encapsulate all the shader code generated for an SDF"""
    def __init__(self, sdf: SDF, prefix=''):
        self.sdf = sdf
        self.prefix = prefix
        
        self.generate_map_func()

        # compile list of params
        # generate basic sdf compute function
        # generate gradient compute function
        # generate edge computing function
        pass

    @staticmethod
    def _traverse(op: ops.OpTree):
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
        leaves = OrderedSet()

        def _traverse_inner(in_count: dict, consumer_nodes: dict, leaves: OrderedSet[ops.OpTree], op: ops.OpTree):
            if isinstance(op, ops.const) or isinstance(op, ops.param): return
            in_count[op] = len(set([a for a in op.args if isinstance(a, ops.OpTree)]))
            is_leaf = True
            for o in op.args:
                if isinstance(o, ops.OpTree):
                    is_leaf = False
                    if o not in consumer_nodes: consumer_nodes[o] = OrderedSet()
                    consumer_nodes[o].add(op)
                    _traverse_inner(in_count, consumer_nodes, leaves, o)
            if is_leaf:
                leaves.add(op)

        _traverse_inner(in_count, consumer_nodes, leaves, op)
        return in_count, consumer_nodes, leaves

    @staticmethod
    def _make_tape(
        in_count: dict[ops.OpTree, int], 
        consumer_nodes: dict[ops.OpBase, OrderedSet[ops.OpTree]], 
        leaves: OrderedSet[ops.OpTree]
    ) -> list[ops.OpTree]:
        """
        Perform a topological sort of the computation graph, generating
        a linear tape
        """
        queue = deque(leaves)
        tape = []
        while len(queue) > 0:
            n = queue.popleft()
            tape.append(n)
            if n in consumer_nodes:
                for pn in consumer_nodes[n]:
                    in_count[pn] -= 1
                    if in_count[pn] == 0:
                        queue.append(pn)
            else:
                assert len(queue) == 0
        return tape
    
    @staticmethod
    def _local_vars_ttl(tape: list[ops.OpTree]) -> dict[ops.OpTree, int]:
        vars_ttl = {}
        for i, l in enumerate(tape):
            for arg in l.args:
                if not isinstance(arg, ops.OpTree): continue
                vars_ttl[arg] = i
        return vars_ttl
    
    def generate_map_func(self):
        p = ops.param(DType(TypeEnum.vec3), '_p', None)
        tree = self.sdf(p)
        in_count, consumer_nodes, leaves = ShaderSDF._traverse(tree)
        tape = ShaderSDF._make_tape(in_count, consumer_nodes, leaves)
        ttl = ShaderSDF._local_vars_ttl(tape)

        lines = []
        var_cnt = 0
        var_map = {} # key is optree, value is name of variable
        var_pool = {} # key is DType, value is deque of variable names
        var_ttl = {} # key is var name, value is (iteration at which it is released, dtype)
        for i, op in enumerate(tape):
            # get local var that we can use
            if op.type not in var_pool: var_pool[op.type] = deque()
            for k in var_ttl:
                if var_ttl[k][0] == i:
                    var_pool[var_ttl[k][1]].add(k)
            if len(var_pool[op.type]) == 0:
                local_var = f'local_var_{var_cnt}'
                var_cnt += 1
                new_var = True
            else:
                local_var = var_pool[op.type].pop()
                new_var = False
            var_ttl[local_var] = (ttl[op], op.type)

            # add line
            lines.append(f'{op.type.get_shader_declaration(local_var) if new_var else local_var} = {op.get_shader_expression(var_map, self.prefix)};')
            # end add line

            var_map[local_var] = op
        self.map_func = '\n'.join(lines)
