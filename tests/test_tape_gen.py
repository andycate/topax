from topax.types import DType, BaseType
import topax.ops as ops
from topax.ops import param
import topax.sdfs as sdfs
from topax._shaders import ShaderSDF

def test_basic_sphere_sdf_optree():
    p = param(DType(BaseType.vec3), '_p', None)
    assert p.implicit == False
    s = sdfs.sphere(0.5)
    tree = s(p)
    assert tree.optype == ops.OpType.SUB
    assert tree.args[0].optype == ops.OpType.LEN
    assert isinstance(tree.args[1], ops.param)
    assert tree.args[1].value == 0.5
    assert tree.args[1].dtype.base == BaseType.float
    assert tree.args[1].implicit == True
    assert tree.dtype.base == BaseType.float

def test_shader_map_gen():
    p = param(DType(BaseType.vec3), '_p', None)
    s = sdfs.sphere(0.5)
    ssdf = ShaderSDF(s, 'test')
    print()
    print(ssdf.map_func)
    print()
    print(ssdf.map_grad_func)
    print()
    print(s._explicit_params)
    print(s._implicit_params)

# def test_basic_sphere_sdf_tape():
#     p = param(TypeEnum.vec3, 'p', None)
#     assert p.implicit == False
#     s = sdfs.sphere(0.5)
#     tree = s(p)
#     in_count, consumer_nodes, leaves = ShaderSDF._traverse(tree)
#     assert len(leaves) == 1
#     assert tree.args[0] in leaves
#     assert len(in_count.keys()) == 2
#     assert in_count[tree] == 1
#     assert in_count[tree.args[0]] == 0

#     tape = ShaderSDF._make_tape(in_count, consumer_nodes, leaves)
#     assert len(tape) == 2
#     assert tape[0] == tree.args[0]
#     assert tape[1] == tree

# def test_basic_sphere_sdf_ttl():
#     p = param('p', None, DType(TypeEnum.vec3))
#     s = sdfs.union(
#         sdfs.sphere(0.5).t(x=0.3),
#         sdfs.sphere(0.5)
#     )
#     tree = s(p)
#     in_count, consumer_nodes, leaves = ShaderSDF._traverse(tree)
#     tape = ShaderSDF._make_tape(in_count, consumer_nodes, leaves)
#     ttl = ShaderSDF._local_vars_ttl(tape)
#     print(ttl)
    


    