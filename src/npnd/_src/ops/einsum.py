import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, List, Union, Any, Optional, Dict

import einsum as einsum_parser

from . import one_hot as one_hot_lib

def einsum_is_identity_spec(spec: einsum_parser.EinsumSpec):
    if len(spec.inputs) != 1:
        return False
    if spec.inputs[0].idxs != spec.output.idxs:
        return False
    assert spec.inputs[0].shape == spec.output.shape, f"{spec}"
    return True

def shape_size(shape) -> int:
    return math.prod(shape)

def reduce_shape(ishape, axes):
    oshape = list(ishape)
    for a in sorted(nonneg(axes, len(ishape)), reverse=True):
        del oshape[a]
    return tuple(oshape)

def make_constant_model(graph_name, output_name, tensor):
    return np.asanyarray(tensor)
    # constant_node = make_constant_node(output_name, tensor)
    # return onnx.helper.make_model(
    #     graph=onnx.helper.make_graph(
    #         name=graph_name,
    #         nodes=[constant_node],
    #         inputs=[],
    #         outputs=[param(output_name, tensor.dtype, tensor.shape)],
    #     )
    # )

Shape = Tuple[int, ...]
OnnxNode = Any

def nonneg(pos, length: int):
    if isinstance(pos, int):
        assert -length <= pos < length
        return pos if pos >= 0 else pos + length
    return map(lambda p: nonneg(p, length), pos)

def omit(seq, *positions):
    for p in sorted(nonneg(positions, len(seq)), reverse=True):
        seq = seq[:p] + seq[p+1:]
    return seq

def squeeze_shape(ishape, axes):
    axes = nonneg(axes, len(ishape))
    return tuple(dim for i, dim in enumerate(ishape) if not (dim == 1 and i in axes))

def unsqueeze_shape(ishape, axes):
    oshape = list(ishape)
    for a in sorted(nonneg(axes, len(ishape) + len(axes))):
        oshape.insert(a, 1)
    return tuple(oshape)

def transpose_seq(ishape, perm):
    return tuple(ishape[a] for a in perm)

def transpose_perm(original_seq, transposed_seq):
    assert sorted(original_seq) == sorted(transposed_seq)
    perm = tuple(original_seq.index(x) for x in transposed_seq)
    assert tuple(transposed_seq) == transpose_seq(original_seq, perm)
    return perm

def get_dim(shape, i):
  if i < 0:
    i += len(shape)
  if i < 0:
    return 0
  if i >= len(shape):
    return len(shape) - 1
  return i

# >>> (np.ones((2,3,5,2,)) @ np.ones((2,3,2,8))).shape
# (2, 3, 5, 8)
# signature is (m?,n),(n,p?)->(m?,p?)
def matmul_shape(shape1, shape2):
  shape1 = tuple(shape1)
  shape2 = tuple(shape2)
  i1 = get_dim(shape1, -1)
  i2 = get_dim(shape2, -2)
  if shape1[i1] != shape2[i2]:
    raise ValueError(f"matmul input operand 1 has a mismatch in its core dimension 0 (size {shape1[i1]} is different from {shape2[i2]})")
  m = shape1[i1-1:i1]
  p = shape2[i2+1:]
  batch_a = shape1[:i1-1]
  batch_b = shape2[:i2]
  batch = np.broadcast_shapes(batch_a, batch_b)
  return batch + m + p

def prod(shape):
    return math.prod(shape) # returns 1 if shape is empty

def reshape_shape(input_shape, output_shape):
  spec = output_shape
  input_shape = list(input_shape)
  output_shape = list(output_shape)
  if spec.count(-1) > 1:
    raise ValueError("can only specify one unknown dimension")
  elif spec.count(-1) == 1:
    # infer unspecified value.
    auto_axis = spec.index(-1)
    auto_shape = list(spec)
    del auto_shape[auto_axis]
    assert prod(auto_shape) != 0
    output_shape[auto_axis] = prod(input_shape) // prod(auto_shape)
  # number of elements in the output shape should match the number of elements in the input shape.
  if prod(input_shape) != prod(output_shape):
    raise ValueError(f"Expected number of elements to be the same. {input_shape} {spec}")
  return output_shape

# def make_constant_node(output_name, tensor) -> OnnxNode:
#     tensor = np.asarray(tensor)
#     return tensor
#     # return onnx.helper.make_node(
#     #     "Constant",
#     #     inputs=[],
#     #     outputs=[output_name],
#     #     value=onnx.helper.make_tensor(
#     #         name=output_name,
#     #         data_type=onnx_type(tensor.dtype),
#     #         dims=tensor.shape,
#     #         vals=tensor.flatten(),
#     #     ),
#     # )
#
# def onnx_helper_make_node(name: str, inputs: List[str], outputs: List[str]):
#         "Squeeze",
#         inputs=[self.oname, axes_name],
#         outputs=[squeeze_name],

@dataclass
class ValueInfo:
    name: str
    dtype: np.dtype
    dims: Shape

@dataclass
class Tensor:
    name: str
    value: np.ndarray

    @classmethod
    def from_array(cls, value: np.ndarray, name: str):
        value = np.asanyarray(value)
        return Tensor(name, value)

@dataclass
class Attribute:
    name: str
    value: Any

@dataclass
class Node:
    op_type: str
    name: str
    input: List[str]
    output: List[str]
    attribute: List[Attribute]

    def get_attribute_value(self, name: str):
        for attr in self.attribute:
            if attr.name == name:
                return attr.value
        raise KeyError(name)

@dataclass
class Graph:
    name: str
    initializer: List[Tensor]
    node: List[Node]
    input: List[ValueInfo]
    output: List[ValueInfo]

    def find(self, name: str):
        for initializer in self.initializer:
            if initializer.name == name:
                return initializer.value
        for input in self.input:
            if input.name == name:
                return input
        for output in self.output:
            if output.name == name:
                return output
        for node in self.node:
            # if node.name == name:
            if name in node.output:
                return node
        raise KeyError(name)

def make_graph(name: str, nodes: List[Node], inputs: List[str], outputs: List[str]):
    initializers = []
    # initializers = [Tensor.from_array(node.value, node.output[0]) for node in nodes if node.op == 'Constant']
    for node in nodes:
        if node.op_type == 'Constant':
            value = node.get_attribute_value('value')
            assert isinstance(value, Tensor)
            initializers.append(value)
    # nodes = [node for node in nodes if node.op != 'Constant']
    return Graph(name, initializers, nodes, inputs, outputs)

@dataclass
class Model:
    graph: Graph

    def run(self, *inputs):
        return run_model(self, *inputs)

def make_model(graph: Graph):
    return Model(graph)

def make_tensor(name: str, value: np.ndarray):
    return Tensor.from_array(value, name)

def make_identity_node(input_name, output_name) -> Node:
    return make_node(
        "Identity",
        output_name,
        inputs=[input_name],
        outputs=[output_name],
    )

def make_constant_model(graph_name: str, output_name: str, tensor):
    tensor = np.asanyarray(tensor)
    constant_node = make_constant_node(output_name, tensor)
    return make_model(
        graph=make_graph(
            name=graph_name,
            nodes=[constant_node],
            inputs=[],
            outputs=[param(output_name, tensor.dtype, tensor.shape)],
        )
    )


def make_tensor_value_info(name: str, dtype, shape: Shape):
    dtype = np.dtype(dtype)
    shape = tuple(shape)
    return ValueInfo(name, dtype, shape)

def param(name: str, dtype, shape) -> ValueInfo:
    # return onnx.helper.make_tensor_value_info(
    #     param_name,
    #     onnx_type(dtype),
    #     shape)
    return make_tensor_value_info(name, dtype, shape)

def make_constant_node(name: str, value: np.ndarray):
    value = np.asanyarray(value)
    return make_node('Constant', name, inputs=[], outputs=[name], value=make_tensor(name, value))

def make_node(op_type: str, name: str, inputs: List[str], outputs: List[str], **attrs):
    return Node(op_type, name=name, input=list(inputs), output=list(outputs), attribute=[Attribute(name, value) for name, value in attrs.items()])

def _asarray(n: Tensor):
    return np.asanyarray(n.value)

import functools

onnx_ops = {}

def onnx_op(name):
    def wrapper(f):
        @functools.wraps(f)
        def func(args, attrs):
            return f(*args, **attrs)
        onnx_ops[name] = func
        return func
    return wrapper

@onnx_op('Identity')
def onnx_identity(value):
    return [value]

@onnx_op('Constant')
def onnx_constant(*, value):
    return [value]

@onnx_op('Reshape')
def onnx_reshape(tensor, shape):
    return [np.asanyarray(tensor).reshape(shape)]

@onnx_op('Mul')
def onnx_mul(A, B):
    return [np.multiply(A, B)]

@onnx_op('MatMul')
def onnx_matmul(A, B):
    return [np.matmul(A, B)]

@onnx_op('Transpose')
def onnx_transpose(tensor, *, perm):
    return [np.transpose(tensor, perm)]

@onnx_op('Squeeze')
def onnx_squeeze(tensor, axes):
    axes = tuple(np.asanyarray(axes).tolist())
    return [np.asanyarray(tensor).squeeze(axes)]

@onnx_op('Unsqueeze')
def onnx_unsqueeze(tensor, axes):
    axes = tuple(np.asanyarray(axes).tolist())
    return [np.expand_dims(tensor, axes)]

@onnx_op('ReduceSum')
def onnx_reducesum(tensor, axis, keepdims):
    axis = tuple(np.asanyarray(axis).tolist())
    return [np.sum(tensor, axis, keepdims=keepdims)]

def interpret_graph(graph: Graph, inputs):
    vals = {node.name: value for node, value in zip(graph.input, inputs)}
    vals.update({n.name: _asarray(n) for n in graph.initializer})
    for op_index, node in enumerate(graph.node):
        args = list(vals[name] for name in node.input)
        attrs = {}
        for attr in node.attribute:
            if isinstance(attr.value, Tensor):
                attrs[attr.name] = attr.value.value
            elif isinstance(attr.value, (list, tuple)):
                attrs[attr.name] = attr.value
            elif isinstance(attr.value, (type(None), bool, int, float)):
                attrs[attr.name] = attr.value
            else:
                raise ValueError("Unknown attr type")
        print(op_index, node.op_type, node.name, *map(np.shape, args), *((attrs,) if attrs else ()), end='\n  => ', flush=True)
        if node.op_type not in onnx_ops:
            print('Unknown')
            breakpoint()
        outputs = onnx_ops[node.op_type](args, attrs)
        print(*map(np.shape, outputs), [np.prod(np.shape(x)) for x in outputs])
        for name, output in zip(node.output, outputs):
            assert isinstance(output, (np.number, np.ndarray))
            vals[name] = output
    return [vals[node.name] for node in graph.output]

def run_model(model: Model, *inputs):
    outputs = interpret_graph(model.graph, inputs)
    assert len(outputs) == 1
    return outputs[0]

@dataclass
class Transform:
    inames: List[str]
    ishapes: List[Shape]
    dtype: Union[np.dtype, type] # e.g. np.type(np.int32) or np.int32
    oname: str
    oshape: Tuple[int, ...]
    nodes: List[Node]

    def graph(self, name: str) -> Graph:
        assert len(self.inames) == len(self.ishapes)
        if len(self.nodes) == 0:
            # Empty graphs don't come compose with onnx.compose, so
            # we insert an Identity node for robustness.
            assert len(self.inames) == 1
            final_oname = f"{name}_out"
            name = self.next_name("identity")
            final_nodes = [make_identity_node(self.inames[0], final_oname)]
        else:
            final_oname = self.oname
            final_nodes = self.nodes
        return make_graph(
            name=name,
            nodes=list(final_nodes),
            inputs=[
                param(iname, self.dtype, ishape)
                    for iname, ishape in sorted(zip(self.inames, self.ishapes))
            ],
            outputs=[param(final_oname, self.dtype, self.oshape)],
        )

    def model(self, graph_name) -> Model:
        graph = self.graph(graph_name)
        return make_model(graph)

    def next_name(self, stem):
        return f"{self.inames[0]}_{stem}_{len(self.nodes)}"

    def squeeze(self, axes):
        oshape = squeeze_shape(self.oshape, axes)
        if tuple(oshape) == tuple(self.oshape):
            return self
        squeeze_name = self.next_name("squeeze")
        if True:
            axes_tensor = np.array(axes, dtype=np.int64)
            axes_name = self.next_name("axes")
            self.nodes.append(make_constant_node(axes_name, axes_tensor))
            self.nodes.append(make_node(
                "Squeeze",
                squeeze_name,
                inputs=[self.oname, axes_name],
                outputs=[squeeze_name],
            ))
        self.oname = squeeze_name
        self.oshape = oshape
        return self

    def unsqueeze(self, axes):
        oshape = unsqueeze_shape(self.oshape, axes)
        if tuple(oshape) == tuple(self.oshape):
            return self
        axes_tensor = np.array(axes, dtype=np.int64)
        axes_name = self.next_name("axes")
        self.nodes.append(make_constant_node(axes_name, axes_tensor))
        unsqueeze_name = self.next_name("unsqueeze")
        self.nodes.append(make_node(
            "Unsqueeze",
            unsqueeze_name,
            inputs=[self.oname, axes_name],
            outputs=[unsqueeze_name],
        ))
        self.oname = unsqueeze_name
        self.oshape = oshape
        return self

    def diagonalize(self, axis1, axis2):
        assert 0 <= axis1 < axis2 < len(self.oshape)
        dim = self.oshape[axis1]
        assert dim == self.oshape[axis2]
        if dim != 1:
            ndim = len(self.oshape)
            indices_shape = self.oshape[:axis1] + (1,) + self.oshape[axis1 + 1:]
            arange = np.arange(dim)
            expanded = np.expand_dims(arange, tuple(range(1, ndim - axis2)))
            indices = np.broadcast_to(expanded, indices_shape)
            hot = one_hot_lib.one_hot(indices, dim, dtype=self.dtype, axis=axis1)
            self.reshape(hot.shape)
            self.mul(self.const(hot))
            self.reducesum([axis1])
            assert(tuple(self.oshape) == tuple(indices_shape))
        return self.squeeze([axis1])

    def reshape(self, shape):
        oshape = reshape_shape(self.oshape, shape)
        if tuple(oshape) == tuple(self.oshape):
            return self
        reshape_name = self.next_name("reshape")
        if True:
            shape_tensor = np.array(shape, dtype=np.int64)
            shape_name = self.next_name("shape")
            self.nodes.append(make_constant_node(shape_name, shape_tensor))
            self.nodes.append(make_node(
                "Reshape",
                reshape_name,
                inputs=[self.oname, shape_name],
                outputs=[reshape_name],
            ))
        self.oname = reshape_name
        self.oshape = oshape
        return self

    def transpose(self, perm):
        assert sorted(perm) == list(range(len(perm)))
        assert len(perm) == len(self.oshape)
        if tuple(perm) == tuple(range(len(self.oshape))):
            return self
        transpose_name = self.next_name("transpose")
        self.nodes.append(make_node(
            "Transpose",
            transpose_name,
            inputs=[self.oname],
            outputs=[transpose_name],
            perm=perm,
        ))
        self.oname = transpose_name
        self.oshape = transpose_seq(self.oshape, perm)
        return self

    def matmul(self, arg):
        self.inames += arg.inames
        self.ishapes += arg.ishapes
        self.nodes += arg.nodes
        matmul_name = self.next_name("matmul")
        self.nodes.append(make_node(
            "MatMul",
            matmul_name,
            inputs=[self.oname, arg.oname],
            outputs=[matmul_name],
        ))
        self.oname = matmul_name
        self.oshape = matmul_shape(self.oshape, arg.oshape)
        return self

    def reducesum(self, axes):
        if len(axes) == 0:
            return self
        sum_name = self.next_name("sum")
        if True:
            axes_tensor = np.array(axes, dtype=np.int64)
            axes_name = self.next_name("axes")
            self.nodes.append(make_constant_node(axes_name, axes_tensor))
            self.nodes.append(make_node(
                "ReduceSum",
                sum_name,
                inputs=[self.oname, axes_name],
                outputs=[sum_name],
                keepdims=0,
            ))
        self.oname = sum_name
        self.oshape = reduce_shape(self.oshape, axes)
        return self

    def const(self, arg: np.ndarray, name: str = None):
        if name is None:
            name = self.next_name("const")
        self.nodes.append(make_constant_node(name, arg))
        return Transform([], [], arg.dtype, name, arg.shape, [])

    def mul(self, arg: Transform):
        mul_shape = np.broadcast_shapes(self.oshape, arg.oshape)
        self.inames += arg.inames
        self.ishapes += arg.ishapes
        self.nodes += arg.nodes
        mul_name = self.next_name("mul")
        if True:
            self.nodes.append(make_node(
                "Mul",
                mul_name,
                inputs=[self.oname, arg.oname],
                outputs=[mul_name],
            ))
        self.oname = mul_name
        self.oshape = mul_shape
        return self

def einsum_squeeze_input(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    axes = [a for a in range(len(shape)) if shape[a] == 1]
    transforms[i].squeeze(axes)
    for a in sorted(axes, reverse=True):
        del idxs[a]
        del shape[a]
    assert tuple(shape) == tuple(transforms[i].oshape)
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_diagonalize_input(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    for a in reversed(range(1, len(idxs))):
        idx = idxs[a]
        b = idxs.index(idx)
        if b != a:
            transforms[i].diagonalize(b, a)
            del idxs[b]
            del shape[b]
            assert tuple(shape) == tuple(transforms[i].oshape)
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_reducesum_input(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i):
    ispec = spec.inputs[i]
    idxs = list(ispec.idxs)
    shape = list(ispec.shape)
    assert len(idxs) == len(set(idxs)), \
        "duplicates indexes after diagonalization pass"
    idxs_in_other_inputs = { idx for ispec in omit(spec.inputs, i) for idx in ispec.idxs }
    idxs_only_in_i = set(idxs) - idxs_in_other_inputs - set(spec.output.idxs)
    axes = [idxs.index(idx) for idx in idxs_only_in_i]
    transforms[i].reducesum(axes)
    for a in sorted(axes, reverse=True):
        del idxs[a]
        del shape[a]
    assert tuple(shape) == tuple(transforms[i].oshape)
    ispec.idxs = idxs
    ispec.shape = tuple(shape)
    return spec, transforms

def einsum_transpose_input(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i, idxs_transposed):
    ispec = spec.inputs[i]
    perm = transpose_perm(ispec.idxs, idxs_transposed)
    transforms[i].transpose(perm)
    ispec.idxs = idxs_transposed
    ispec.shape = transpose_seq(ispec.shape, perm)
    assert transforms[i].oshape == ispec.shape
    return spec, transforms

def einsum_contract_inputs(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i, j):
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    ij_idxs = set(i_ispec.idxs) & set(j_ispec.idxs)
    idxs_in_other_inputs = {
        idx for ispec in omit(spec.inputs, i, j) for idx in ispec.idxs
    }
    idxs2keep = idxs_in_other_inputs.union(spec.output.idxs)
    idxs2reduce = ij_idxs.difference(idxs2keep)
    if len(idxs2reduce) == 0:
        return einsum_mul_inputs(spec, transforms, i, j)
    else:
        return einsum_matmul_inputs(spec, transforms, idxs2reduce, i, j)

# matmul is an optimization of mul followed by reducesum:
# def einsum_matmul_inputs(spec, transforms, idxs2reduce, i, j):
#   spec, transforms = einsum_mul_inputs(spec, transforms, i, j)
#   i -= j < i # j's removal may shift i one position to the left
#   return einsum_reducesum_input(spec, transforms, i)
#
def einsum_matmul_inputs(spec: einsum_parser.EinsumSpec, transforms: List[Transform], idxs2reduce, i, j):
    # We assume that each of i and j have no repeated or reducible indexes.
    # (Any repeated or reducible indexes in each input were removed in
    # einsum_diagonalize_input() and einsum_reducesum_input() up front
    # and einsum_contract_inputs() doesn't produce any repeated or reducible
    # indexes.)
    #
    # Under this assumption the indexes in i and j fall in 4 buckets:
    # 1. Those that are reducible after or during contraction, namely those
    #    not in the output or any other remaining inputs. These appear in
    #    both i and j (as we assume no reducible indexes in each input).
    # 2. The other indexes that appear in both i and j,
    # 3. The indexes that appear in i and not in j.
    # 4. The indexes that appear in j and not in i.
    #
    # The indexes in the output of the contraction are the disjoint
    # union of buckets 2, 3, 4.

    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    i_idxs = i_ispec.idxs
    j_idxs = j_ispec.idxs
    ij_idxs = set(i_idxs) & set(j_idxs)
    ij_keep_idxs = [idx for idx in i_idxs if idx in ij_idxs - idxs2reduce]
    ij_reduce_idxs = [idx for idx in i_idxs if idx in idxs2reduce]
    i_idxs_unshared = [idx for idx in i_idxs if idx not in ij_idxs]
    j_idxs_unshared = [idx for idx in j_idxs if idx not in ij_idxs]

    i_idxs_transposed = ij_keep_idxs + i_idxs_unshared + ij_reduce_idxs
    spec, transforms = einsum_transpose_input(spec, transforms, i, i_idxs_transposed)
    j_idxs_transposed = ij_keep_idxs + ij_reduce_idxs + j_idxs_unshared
    spec, transforms = einsum_transpose_input(spec, transforms, j, j_idxs_transposed)
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]

    ij_keep_shape = j_ispec.shape[0:len(ij_keep_idxs)]
    ij_reduce_shape = j_ispec.shape[len(ij_keep_idxs):len(ij_idxs)]
    j_unshared_shape = j_ispec.shape[len(ij_idxs):]
    i_unshared_shape = i_ispec.shape[len(ij_keep_idxs):][:len(i_idxs_unshared)]
    ij_reduce_size = prod(ij_reduce_shape)
    i_unshared_size = prod(i_unshared_shape)
    j_unshared_size = prod(j_unshared_shape)

    transforms[i].reshape(ij_keep_shape + (i_unshared_size, ij_reduce_size))
    assert len(transforms[i].oshape) == len(ij_keep_idxs) + 2
    transforms[j].reshape(ij_keep_shape + (ij_reduce_size, j_unshared_size))
    assert len(transforms[j].oshape) == len(ij_keep_idxs) + 2
    transforms[i].matmul(transforms[j])
    assert transforms[i].oshape == ij_keep_shape + (i_unshared_size, j_unshared_size)
    final_shape = ij_keep_shape + i_unshared_shape + j_unshared_shape
    transforms[i].reshape(final_shape)
    i_ispec.shape = final_shape
    i_ispec.idxs = ij_keep_idxs + i_idxs_unshared + j_idxs_unshared
    assert len(i_ispec.idxs) == len(i_ispec.shape), f"{i_ispec}"
    del transforms[j]
    del spec.inputs[j]
    return spec, transforms

def einsum_mul_inputs(spec: einsum_parser.EinsumSpec, transforms: List[Transform], i, j):
    i_ispec = spec.inputs[i]
    j_ispec = spec.inputs[j]
    i_idxs = i_ispec.idxs
    j_idxs = j_ispec.idxs
    ij_idxs = set(i_idxs) & set(j_idxs)

    # transpose j so it ends with the idxs that also occur in i, in the same order
    j_idxs_unshared = [idx for idx in j_idxs if idx not in ij_idxs]
    j_idxs_shared = [idx for idx in i_idxs if idx in ij_idxs]
    j_idxs_transposed = j_idxs_unshared + j_idxs_shared
    spec, transforms = einsum_transpose_input(spec, transforms, j, j_idxs_transposed)
    j_ispec = spec.inputs[j]

    # unsqueeze j so ends with all i's idxs, in the same order
    axes = [a for a in range(-len(i_idxs), 0) if i_idxs[a] not in ij_idxs]
    j_idxs_unsqueezed = j_idxs_unshared + i_idxs
    transforms[j].unsqueeze(axes)
    j_ispec.idxs = j_idxs_unsqueezed
    j_ispec.shape = unsqueeze_shape(j_ispec.shape, axes)
    assert j_ispec.shape == transforms[j].oshape
    assert len(j_ispec.shape) == len(j_idxs_unsqueezed)

    # mul() broadcasts i to unsqueezed j's rank
    transforms[i].mul(transforms[j])
    i_ispec.idxs = j_idxs_unsqueezed
    i_ispec.shape = np.broadcast_shapes(i_ispec.shape, j_ispec.shape)
    assert i_ispec.shape == transforms[i].oshape
    assert len(i_ispec.shape) == len(j_idxs_unsqueezed)
    del transforms[j]
    del spec.inputs[j]
    return spec, transforms

def einsum_finalize(spec: einsum_parser.EinsumSpec, transform: Transform):
    assert len(spec.inputs) == 1
    ispec = spec.inputs[0]
    in_idxs = set(ispec.idxs)
    out_idxs = spec.output.idxs
    assert in_idxs.issubset(set(out_idxs)), f"{in_idxs},{out_idxs}"
    assert all(idx in in_idxs or spec.idxs_map[idx] == 1 for idx in out_idxs)
    if einsum_is_identity_spec(spec):
        # The equation is the identity transformation.
        return transform
    squeezed_out_idxs = [idx for idx in out_idxs if idx in in_idxs]
    perm = tuple(ispec.idxs.index(idx) for idx in squeezed_out_idxs)
    transform.transpose(perm)
    axes = [a for a in range(len(out_idxs)) if out_idxs[a] not in in_idxs]
    return transform.unsqueeze(axes)

def make_identity_transform(dtype, shape, iname):
    return Transform([iname], [shape], dtype, iname, shape, [])

def einsum_decomposed_model(equation, ishapes, dtype):
    spec = einsum_parser.einsum_spec(equation, ishapes)

    # In two cases the output is just zeros or empty:
    # (1) empty if there are any 0 dims in the output shape,
    # (2) zeros if there are any 0 dims in any input shape
    # (because they either occur in the output, which would be
    # empty, or will be eliminated by ReduceSum and become zeros).
    oshape = spec.output.shape
    if any(shape_size(shape) == 0 for shape in ishapes + [oshape]):
        tensor = np.zeros(oshape, dtype=dtype)
        return make_constant_model('einsum_constant', "out", tensor), tuple(spec.output.shape)

    # Each transform is either an onnx model transforming the input
    # at that position or just a string with the name of the input
    # which represents the identity transformation.
    ninputs = len(ishapes)
    assert ninputs <= 100 # for convenience to keep input names short
    in_name = lambda i: "in%02d" % i # sortable names for i < 100
    transforms = [
        make_identity_transform(dtype, ishapes[i], in_name(i))
        for i in range(ninputs)
    ]

    for i in range(ninputs):
        # squeezing avoids broadcasting in contractions amd it can potentially
        # be an optimization to skip some diagonalizations and reducesums and
        # axes to juggle in contractions
        spec, transforms = einsum_squeeze_input(spec, transforms, i)
        spec, transforms = einsum_diagonalize_input(spec, transforms, i)

    # einsum_squeeze_input() must be done on all inputs before
    # einsum_reducesum_input() on any input, because a squeeze of
    # a later input can enable reducesum of an earlier input
    for i in range(ninputs):
        spec, transforms = einsum_reducesum_input(spec, transforms, i)

    # TODO: optimize the contraction order
    while len(transforms) > 1:
        spec, transforms = einsum_contract_inputs(spec, transforms, 0, 1)

    transform = einsum_finalize(spec, transforms[0])
    return transform.model(f"einsum({equation})"), tuple(transform.oshape)



def einsum(equation: str, *tensors: np.ndarray) -> np.ndarray:
    ishapes = [np.shape(tensor) for tensor in tensors]
    spec = einsum_parser.einsum_spec(equation, ishapes)
    return einsum_parser.einsum_execute(spec, list(tensors))

def einsum_model(equation: str, ishapes: Shape, dtype=np.float64):
    model, oshape = einsum_decomposed_model(equation, ishapes, dtype)
    return model, oshape
    # ishapes = [np.shape(tensor) for tensor in tensors]
    # spec = einsum_parser.einsum_spec(equation, ishapes)
    # return einsum_parser.einsum_execute(spec, list(tensors))

def einsum_model_test():
    import npnd
    for equation, ishapes in [
            ("ii->i", [(0,0)]),
            ("ii", [(0,0)]),
            ("ij,jk", [(0,2),(2,2)]),
            ("ij,jk->k", [(0,2),(2,2)]),
            ("i", [(2,)]),
            ("...", [(2,3,4)]),
            ("ij...k->...ijk", [(2,3,4)]),
            # squeezes axes s,t,u:
            ("sij->ij", [(1,2,3)]),
            ("isj->ij", [(2,1,3)]),
            ("ijs->ij", [(2,3,1)]),
            ("sitju->ij", [(1,2,1,3,1)]),
            # diagonalize axes s,t:
            ("ss->s", [(2,2)]),
            ("ssuu->su", [(2,2,3,3)]),
            ("sss->s", [(2,2,2)]),
            ("iss->is", [(3,2,2)]),
            ("sis->is", [(2,3,2)]),
            ("ssi->si", [(2,2,3)]),
            # reducesum axes s,t,u:
            ("sij->ij", [(4,2,3)]),
            ("isj->ij", [(2,4,3)]),
            ("ijs->ij", [(2,3,4)]),
            ("sitju->ij", [(4,2,5,3,6)]),
            # transpose:
            ("ij->ji", [(2,3)]),
            ("ijk->jik", [(2,3,4)]),
            ("ijk->jki", [(2,3,4)]),
            ("ijk->kji", [(2,3,4)]),
            ("ijk->ijk", [(2,3,4)]),
            ("ijk->ikj", [(2,3,4)]),
            ("ijk->kij", [(2,3,4)]),
            # unsqueeze:
            ("ij", [(1,2)]),
            ("ij->ji", [(1,2)]),
            ("ij", [(1,1)]),
            ("ij->ji", [(1,1)]),
            ("ghijk,ghjkm->ghim", [(1,5,2,1,3),(6,1,3,1,4)]),
            # matmul:
            ("ij,j", [(2,3),(3,)]),
            ("i,i", [(2,),(2,)]),
            ("ij,ij", [(2,3),(2,3)]),
            ("ij,ji", [(2,3),(3,2)]),
            ("ij,jk", [(2,3),(3,4)]),
            ("hij,hjk", [(5,2,3),(5,3,4)]),
            ("ghijk,ghjkm", [(6,5,2,3,3),(6,5,3,3,4)]),
            ("ghijk,ghjkm,gh", [(6,5,2,3,3),(6,5,3,3,4),(6,5)]),
            ("ghijk,ghjkm->ghim", [(6,5,2,3,3),(6,5,3,3,4)]),
            # outer:
            ("i,j->ij", [(3,), (4,)]),
            ("i,j->i", [(3,), (4,)]),
            ("i,j->j", [(3,), (4,)]),
            ("i,j->", [(3,), (4,)]),
        ]:
        model, oshape = einsum_model(equation, ishapes)
        values = [npnd.values(shape) for shape in ishapes]
        expected = np.einsum(equation, *values)
        result = model.run(*values)
        assert np.allclose(expected, result)
        assert tuple(oshape) == np.shape(expected)
