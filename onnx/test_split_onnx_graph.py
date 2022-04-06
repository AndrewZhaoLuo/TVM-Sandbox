"""Example of splitting ONNX graph manually

https://github.com/onnx/onnx/blob/main/docs/IR.md
"""

import glob
import os
from collections import defaultdict, deque
from more_itertools import consumer

import onnxruntime as ort

import onnx.utils
from onnx import shape_inference
from onnx.tools import update_model_dims
from six import string_types
import tvm
from tvm import relay
from onnx import helper
from typing import *

OP_TYPES_OF_INTEREST = {
    "Conv",
    "MatMul",
    "Gemm",
    "ConvInteger",
    "ConvTranspose",
    "MatMulInteger",
    "QLinearConv",
    "QLinearMatMul",
    "RNN",
    "LSTM",
}

# Map of op names to argument indices which may induce dynamism in model
DYNAMIC_OPS = {"Reshape": (1,)}

import numpy as np


def update_inputs_outputs_dims(
    model, input_dims, output_dims
):  # type: (ModelProto, Dict[Text, List[Any]], Dict[Text, List[Any]]) -> ModelProto
    """
    Copy from from onnx.tools import update_model_dims with one diff

    This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    Example. if we have the following shape for inputs and outputs:
            shape(input_1) = ('b', 3, 'w', 'h')
            shape(input_2) = ('b', 4)
            and shape(output)  = ('b', 'd', 5)

        The parameters can be provided as:
            input_dims = {
                "input_1": ['b', 3, 'w', 'h'],
                "input_2": ['b', 4],
            }
            output_dims = {
                "output": ['b', -1, 5]
            }

        Putting it together:
            model = onnx.load('model.onnx')
            updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
            onnx.save(updated_model, 'model.onnx')
    """
    dim_param_set = set()  # type: Set[Text]

    def init_dim_param_set(
        dim_param_set, value_infos
    ):  # type: (Set[Text], List[ValueInfoProto]) -> None
        for info in value_infos:
            shape = info.type.tensor_type.shape
            for dim in shape.dim:
                if dim.HasField("dim_param"):
                    dim_param_set.add(dim.dim_param)  # type: ignore

    init_dim_param_set(dim_param_set, model.graph.input)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.output)  # type: ignore
    init_dim_param_set(dim_param_set, model.graph.value_info)  # type: ignore

    def update_dim(
        tensor, dim, j, name
    ):  # type: (ValueInfoProto, Any, int, Text) -> None
        dim_proto = tensor.type.tensor_type.shape.dim[j]
        if isinstance(dim, int):
            if dim >= 0:
                if dim_proto.HasField("dim_value") and dim_proto.dim_value != dim:
                    raise ValueError(
                        "Unable to set dimension value to {} for axis {} of {}. Contradicts existing dimension value {}.".format(
                            dim, j, name, dim_proto.dim_value
                        )
                    )
                dim_proto.dim_value = dim
            else:
                generated_dim_param = name + "_" + str(j)
                if generated_dim_param in dim_param_set:
                    raise ValueError(
                        "Unable to generate unique dim_param for axis {} of {}. Please manually provide a dim_param value.".format(
                            j, name
                        )
                    )
                dim_proto.dim_param = generated_dim_param
        elif isinstance(dim, string_types):
            dim_proto.dim_param = dim
        else:
            raise ValueError(
                "Only int or str is accepted as dimension value, incorrect type: {}".format(
                    type(dim)
                )
            )

    for input in model.graph.input:
        input_name = input.name
        if input_name in input_dims:
            input_dim_arr = input_dims[input_name]
            for j, dim in enumerate(input_dim_arr):
                update_dim(input, dim, j, input_name)

    for output in model.graph.output:
        output_name = output.name
        if output_name in output_dims:
            output_dim_arr = output_dims[output_name]
            for j, dim in enumerate(output_dim_arr):
                update_dim(output, dim, j, output_name)

    onnx.checker.check_model(model)
    return model


def set_unique_names(onnx_model):
    i = 0
    for node in onnx_model.graph.node:
        if node.name == "":  # no name, create a unique one
            node.name = f"assigned_name_{str(node.op_type)}_{i}"
            i += 1
    return onnx_model


def load_model(
    model_path: str,
    input_shapes: Optional[Dict[str, List]] = None,
    output_shapes: Optional[Dict[str, List]] = None,
) -> onnx.ModelProto:
    """Load the given model, updating shapes as need be."""
    onnx_model = onnx.load(model_path)
    onnx_model = set_unique_names(onnx_model)

    if input_shapes is not None and output_shapes is not None:
        # This doesn't work too well with some models. Future Idea: run model to get shapes
        # Set shapes so each component is able to be calculated with real shapes
        onnx_model = update_inputs_outputs_dims(
            onnx_model,
            input_shapes,
            output_shapes,
        )
    onnx_model = shape_inference.infer_shapes(onnx_model)
    return onnx_model


def get_node_and_consumer_map(
    onnx_model: onnx.ModelProto,
) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, Set[str]]]:
    """Get a map of node names --> onnx nodes and consumer map."""
    # Map of name of node --> ONNX node
    node_map = {}

    # Map of names --> names of nodes who consume name
    consumer_map = defaultdict(set)

    for node in onnx_model.graph.node:
        assert node.name not in node_map
        node_map[node.name] = node
        for input_name in node.input:
            consumer_map[input_name].add(node.name)

    return node_map, consumer_map


def get_sections(
    onnx_model: onnx.ModelProto,
    op_types_of_interest: Set[str],
    max_ops_of_interest: int,
) -> List[List[str]]:
    """Divides up the onnx model graph into subgraphs such that each subgraph will have
    up to `max_ops_of_interest` number of operations belonging to `op_types_of_interest`

    Each section is non-overlapping except for inputs/outputs. In general the outputs
    of one graph will be the input to another graph.

    Returns a list of sections, where each section is a list of named nodes which belong
    to a subgraph.
    """
    node_map, _ = get_node_and_consumer_map(onnx_model)

    # Tuple of input names, output names for extraction
    sections = []
    cur_section = []
    ops_of_interest = 0

    visited = set()
    search_space = deque()
    for node in onnx_model.graph.node:
        if node.name in visited:
            continue

        search_space.appendleft(node.name)

        while len(search_space) > 0:
            cur_node_name = search_space.popleft()
            if cur_node_name in visited:
                continue
            visited.add(cur_node_name)

            if cur_node_name not in node_map:
                # TODO: figure out what this means
                continue

            cur_node = node_map[cur_node_name]

            for output_name in cur_node.output:
                search_space.appendleft(output_name)

            if cur_node.op_type in op_types_of_interest:
                ops_of_interest += 1
                if ops_of_interest > max_ops_of_interest:
                    sections.append(cur_section)
                    ops_of_interest = 1
                    cur_section = []

            cur_section.append(cur_node.name)

    if len(cur_section) > 0:
        sections.append(cur_section)

    return sections


def extract_sections(
    extractor: onnx.utils.Extractor,
    sections: List[List[str]],
    node_map: Dict[str, onnx.NodeProto],  # map of node name to the nodes
    default_tensors: Set[str],  # set of names in initializer list of onnx graph
    consumer_graph: Dict[str, Set[str]],
) -> List[onnx.ModelProto]:
    """From a list of sections, extracts subgraphs containing those sections

    And returns a corresponding list of model protos which are topographical order
    w.r.t. their output nodes.
    """
    ret = []
    for section in sections:
        input_nodes = set()
        output_nodes = set()
        for node_name in section:
            node = node_map[node_name]
            for node_input in node.input:
                input_nodes.add(node_input)
            for node_output in node.output:
                output_nodes.add(node_output)

        # If all nodes are consumed in this seciton
        common_nodes = input_nodes.intersection(output_nodes)

        # The set of all output whose connections are not strictly in this section
        # (We want these to represent true topology of graph)
        loose_output_nodes = set()
        for node in common_nodes.intersection(output_nodes):
            # check if all consumers of this node are in the graph
            global_consumers = consumer_graph[node]
            all_nodes_in_graph = set(section).union(input_nodes).union(output_nodes)
            consumers_in_graph = global_consumers.intersection(all_nodes_in_graph)
            if len(consumers_in_graph) != len(global_consumers):
                loose_output_nodes.add(node)

        input_nodes = input_nodes - common_nodes - default_tensors
        output_nodes = output_nodes - common_nodes - default_tensors
        output_nodes = output_nodes.union(loose_output_nodes)

        """
        # Debug info 
        print("Section:", section)
        print("Inputs:")
        for input_node in input_nodes:
            print(f"\t{input_node}")
        print("Outputs:")
        for output_node in output_nodes:
            print(f"\t{output_node}")
        print()
        """

        section = extractor.extract_model(input_nodes, output_nodes)
        ret.append(section)
    return ret


def run_model(
    onnx_model: onnx.ModelProto, input_values: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    session = ort.InferenceSession(onnx_model.SerializeToString())
    result = session.run(None, input_values)

    output_values = {}
    for i, output_node in enumerate(onnx_model.graph.output):
        name = output_node.name
        tensor = result[i]
        output_values[name] = tensor
    return output_values


def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
) -> onnx.TensorProto:
    """Makes an initizlier proto."""
    DTYPE_MAP = {
        "float16": onnx.TensorProto.FLOAT16,
        "float32": onnx.TensorProto.FLOAT,
        "float64": onnx.TensorProto.DOUBLE,
        "int8": onnx.TensorProto.INT8,
        "int16": onnx.TensorProto.INT16,
        "int32": onnx.TensorProto.INT32,
        "int64": onnx.TensorProto.INT64,
        "uint8": onnx.TensorProto.UINT8,
        "uint16": onnx.TensorProto.UINT16,
        "uint32": onnx.TensorProto.UINT32,
        "uint64": onnx.TensorProto.UINT64,
    }
    # (TensorProto)
    initializer_tensor = helper.make_tensor(
        name=name,
        data_type=DTYPE_MAP[str(tensor_array.dtype)],
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )

    return initializer_tensor


def simplify_model(
    onnx_model: onnx.ModelProto,
    input_values: Dict[str, np.ndarray],
    dynamic_ops: Dict[str, List[int]],
) -> Tuple[onnx.ModelProto, Dict[str, np.ndarray]]:
    """Runs the onnx model while also mutating it to make certain dynamic ops non-dynamic."""
    node_map, _ = get_node_and_consumer_map(onnx_model)

    # Get list of inputs which should be frozen (set as constants)
    # Because they feed into a dynamic op's argument which causes dynamism
    # TODO: propagate dynamism argument to adjacent nodes
    frozen_input_values = set()
    for k, v in node_map.items():
        if v.op_type in dynamic_ops.keys():
            bad_arg_pos = dynamic_ops[v.op_type]
            for arg_pos in bad_arg_pos:
                input_name = v.input[arg_pos]
                if input_name in input_values.keys():
                    frozen_input_values.add(input_name)

    for k, v in list(input_values.items()):
        if k not in frozen_input_values:
            input_values[k] = np.zeros_like(v)
        else:
            tensor_proto = create_initializer_tensor(k, input_values[k])
            input_values.pop(k)
            onnx_model.graph.initializer.append(tensor_proto)
            # Now it is initializer, remove from inputs
            for i, input_node in enumerate(list(onnx_model.graph.input)):
                if input_node.name == k:
                    onnx_model.graph.input.pop(i)
                    break

    return onnx_model, input_values


def split_model(
    onnx_model: onnx.ModelProto,
    input_tensors: Dict[str, np.ndarray],
    op_types_of_interest: Set[str] = OP_TYPES_OF_INTEREST,
    max_ops_of_interest: int = 1,
    dynamic_ops: Dict[str, List[int]] = DYNAMIC_OPS,
) -> List[onnx.ModelProto]:
    extractor = onnx.utils.Extractor(onnx_model)
    # Constant tensors
    default_tensors = {t.name for t in onnx_model.graph.initializer}

    node_map, consumer_graph = get_node_and_consumer_map(onnx_model)

    # Tuple of input names, output names for extraction
    sections = get_sections(onnx_model, op_types_of_interest, max_ops_of_interest)
    print(f"Extracted {len(sections)} sections!")

    onnx_subgraphs = extract_sections(
        extractor, sections, node_map, default_tensors, consumer_graph
    )
    print(f"Extracted {len(onnx_subgraphs)} subgraphs!")

    # Cache tensors we might need them later
    tensors = dict(input_tensors)
    result = []

    # Run the model to infer shape and set certain inputs to dynamics ops as constant
    for i, onnx_sub_model in enumerate(onnx_subgraphs):
        print(f"Simplifying model {i + 1} / {len(onnx_subgraphs)}")
        input_names = [node.name for node in onnx_sub_model.graph.input]
        input_tensors = {k: tensors[k] for k in input_names}
        onnx_sub_model, input_tensors = simplify_model(
            onnx_sub_model, input_tensors, dynamic_ops
        )
        output_tensors = run_model(onnx_sub_model, input_tensors)
        tensors.update(output_tensors)
        result.append(onnx_sub_model)
    return result


def compare_onnxrt_to_tvm(onnx_model, input_values):
    output_tensors = run_model(onnx_model, input_values)
    ordered_tensors = [output_tensors[node.name] for node in onnx_model.graph.output]

    # Run TVM session
    mod, _ = relay.frontend.from_onnx(
        onnx_model,
        shape={k: v.shape for k, v in input_values.items()},
        dtype={k: str(v.dtype) for k, v in input_values.items()},
        freeze_params=True,
    )
    format_str = str(mod)
    lines = format_str.split("\n")
    for i in range(len(lines)):
        lines[i] = "\t" + lines[i]
    format_str = "\n".join(lines)
    print(format_str)

    vm_exe = relay.create_executor("vm", mod=mod)
    tvm_result = vm_exe.evaluate()(**input_values)
    if isinstance(tvm_result, tvm.runtime.container.ADT):
        tvm_result = [r.asnumpy() for r in tvm_result]
    else:
        tvm_result = [tvm_result.asnumpy()]

    assert len(tvm_result) == len(ordered_tensors)
    for tvm_result, onnx_result in zip(tvm_result, ordered_tensors):
        np.testing.assert_allclose(tvm_result, onnx_result, rtol=0.05, atol=1e-3)

    return output_tensors


def extract_model(
    model_path: str,
    input_shapes: Dict[str, List[int]],
    input_dtypes: Dict[str, str],
    output_shapes: Dict[str, List[int]],
    op_types_of_interest: Set[str] = OP_TYPES_OF_INTEREST,
    dir: str = "onnx_segments",
):
    os.makedirs(dir, exist_ok=True)
    onnx_model = load_model(model_path, input_shapes, output_shapes)
    model_names = []
    input_tensors = {
        name: np.zeros(input_shapes[name]).astype(input_dtypes[name])
        for name in input_shapes.keys()
    }
    onnx_sub_models = split_model(
        onnx_model, input_tensors, op_types_of_interest=op_types_of_interest
    )

    base_file_name = os.path.basename(model_path).split(".onnx")[0]

    for i, segment in enumerate(onnx_sub_models):
        main_ops = [
            node.op_type
            for node in segment.graph.node
            if node.op_type in op_types_of_interest
        ]
        output_path = os.path.join(
            dir, f"{base_file_name}-segment-{i}-{''.join(main_ops)}.onnx"
        )
        onnx.save(segment, output_path)
        model_names.append(output_path)
    return model_names


def run_examination(
    input_shapes: Dict[str, List[int]],
    input_dtypes: Dict[str, str],
    model_path: str,
):
    input_names = list(input_shapes.keys())
    models = extract_model(model_path, input_shapes, input_dtypes, output_shapes={})
    tensors = {
        name: np.zeros(input_shapes[name]).astype(input_dtypes[name])
        for name in input_names
    }
    for model_path in models:
        print(model_path)
        onnx_model = load_model(model_path)
        input_names = [node.name for node in onnx_model.graph.input]
        input_tensors = {k: tensors[k] for k in input_names}
        output_tensors = compare_onnxrt_to_tvm(onnx_model, input_tensors)

        tensors.update(output_tensors)


def try_bertsquad():
    input_shapes = {
        "unique_ids_raw_output___9:0": [1],
        "segment_ids:0": [1, 256],
        "input_mask:0": [1, 256],
        "input_ids:0": [1, 256],
    }
    input_dtypes = {
        "unique_ids_raw_output___9:0": "int64",
        "segment_ids:0": "int64",
        "input_mask:0": "int64",
        "input_ids:0": "int64",
    }
    run_examination(input_shapes, input_dtypes, "models/bertsquad-8.onnx")


def try_ssd():
    input_shapes = {"image": [1, 3, 1200, 1200]}
    input_dtypes = {"image": "float32"}
    run_examination(input_shapes, input_dtypes, "models/ssd-10.onnx")


def try_yolov2():
    input_shapes = {"input.1": [1, 3, 416, 416]}
    input_dtypes = {"input.1": "float32"}
    run_examination(input_shapes, input_dtypes, "models/yolov2-coco-9.onnx")


def try_shufflenet():
    input_shapes = {"gpu_0/data_0": [1, 3, 224, 224]}
    input_dtypes = {"gpu_0/data_0": "float32"}
    run_examination(input_shapes, input_dtypes, "models/shufflenet-9.onnx")


def try_resnet():
    input_shapes = {"data": [1, 3, 224, 224]}
    input_dtypes = {"data": "float32"}
    run_examination(input_shapes, input_dtypes, "models/resnet50-v1-7.onnx")

if __name__ == "__main__":
    # try_bertsquad()
    # try_ssd()
    # try_yolov2()
    # try_shufflenet()
    try_resnet()