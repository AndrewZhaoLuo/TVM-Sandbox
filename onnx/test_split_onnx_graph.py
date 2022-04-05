"""Example of splitting ONNX graph manually

https://github.com/onnx/onnx/blob/main/docs/IR.md
"""

import os
from collections import defaultdict, deque

import onnxruntime as ort

import onnx.utils
from onnx import shape_inference
from onnx.tools import update_model_dims
from six import string_types
import tvm
from tvm import relay
from onnx import helper

OP_TYPES_OF_INTEREST = {"Conv", "MatMul"}

# These ops add dynamism typically and should be avoided!
OP_TYPES_IGNORE = {"Reshape"}

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


def load_model(model_path, input_shapes=None, output_shapes=None):
    onnx_model = onnx.load(model_path)

    if input_shapes is not None and output_shapes is not None:
        # This doesn't work too well with some models. Future Idea: run model to get shapes
        # Set shapes so each component is able to be calculated with real shapes
        onnx_model = update_model_dims.update_inputs_outputs_dims(
            onnx_model,
            input_shapes,
            output_shapes,
        )
    onnx_model = shape_inference.infer_shapes(onnx_model)
    return onnx_model


def get_node_and_consumer_map(onnx_model):
    # Map of name of node --> ONNX node
    node_map = {}

    # Map of names --> names of nodes who consume name
    consumer_map = defaultdict(set)

    for node in onnx_model.graph.node:
        node_map[node.name] = node
        for input_name in node.input:
            consumer_map[input_name].add(node.name)

    return node_map, consumer_map


def split_model(
    onnx_model, op_types_of_interest=OP_TYPES_OF_INTEREST, max_ops_of_interest=1
):
    extractor = onnx.utils.Extractor(onnx_model)
    # Constant tensors
    default_tensors = {t.name for t in onnx_model.graph.initializer}

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
                    print(len(sections))

            cur_section.append(cur_node.name)

    if len(cur_section) > 0:
        sections.append(cur_section)

    ret = []
    for i, section in enumerate(sections):
        input_nodes = set()
        output_nodes = set()
        for node_name in section:
            node = node_map[node_name]
            for node_input in node.input:
                input_nodes.add(node_input)
            for node_output in node.output:
                output_nodes.add(node_output)

        common_nodes = input_nodes.intersection(output_nodes)

        input_nodes = input_nodes - common_nodes - default_tensors
        output_nodes = output_nodes - common_nodes - default_tensors
        print("Section:", section)
        print("Inputs:")
        for input_node in input_nodes:
            print(f"\t{input_node}")
        print("Outputs:")
        for output_node in output_nodes:
            print(f"\t{output_node}")
        print()
        section = extractor.extract_model(input_nodes, output_nodes)
        ret.append(section)
    return ret


def extract_model():
    onnx_model = load_model(
        "models/bertsquad-8.onnx",
        {
            "unique_ids_raw_output___9:0": [1],
            "segment_ids:0": [1, 256],
            "input_mask:0": [1, 256],
            "input_ids:0": [1, 256],
        },
        {"unstack:1": [1, 256], "unstack:0": [1, 256], "unique_ids:0": [1]},
    )
    model_names = []
    model_segments = split_model(onnx_model)
    for i, segment in enumerate(model_segments):
        onnx.save(segment, f"bertsquad-segment{i}.onnx")
        model_names.append(f"bertsquad-segment{i}.onnx")
    return model_names


def create_initializer_tensor(
    name: str,
    tensor_array: np.ndarray,
) -> onnx.TensorProto:

    DTYPE_MAP = {
        "float32": onnx.TensorProto.FLOAT,
        "float64": onnx.TensorProto.DOUBLE,
        "int8": onnx.TensorProto.INT8,
        "int64": onnx.TensorProto.INT64,
    }
    # (TensorProto)
    initializer_tensor = helper.make_tensor(
        name=name,
        data_type=DTYPE_MAP[str(tensor_array.dtype)],
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )

    return initializer_tensor


def run_segment(onnx_model, input_values, SPECIAL_OPERATIONS={"Reshape": (1,)}):
    node_map, _ = get_node_and_consumer_map(onnx_model)

    frozen_input_values = set()

    # Only ops which are near to the thing
    for k, v in node_map.items():
        if v.op_type in SPECIAL_OPERATIONS.keys():
            bad_arg_pos = SPECIAL_OPERATIONS[v.op_type]
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

    # Run onnx session
    session = ort.InferenceSession(onnx_model.SerializeToString())
    result = session.run(None, input_values)
    output_values = {}
    for i, output_node in enumerate(onnx_model.graph.output):
        name = output_node.name
        tensor = result[i]
        output_values[name] = tensor

    # Run TVM session
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape={k: v.shape for k, v in input_values.items()},
        dtype={k: str(v.dtype) for k, v in input_values.items()},
        freeze_params=True,
    )
    return output_values


if __name__ == "__main__":
    # extract_model()

    models = os.listdir()
    models = [m for m in models if "bertsquad-segment" in m]
    models.sort(key=lambda x: int(x.split("segment")[-1].split(".")[0]))

    print(models)

    first_model = models[0]
    all_shapes = {
        "unique_ids_raw_output___9:0": [1],
        "segment_ids:0": [1, 256],
        "input_mask:0": [1, 256],
        "input_ids:0": [1, 256],
    }
    all_dtypes = {
        "unique_ids_raw_output___9:0": "int64",
        "segment_ids:0": "int64",
        "input_mask:0": "int64",
        "input_ids:0": "int64",
    }
    input_names = [
        "unique_ids_raw_output___9:0",
        "segment_ids:0",
        "input_mask:0",
        "input_ids:0",
    ]
    tensors = {
        name: np.zeros(all_shapes[name]).astype(all_dtypes[name])
        for name in input_names
    }
    for model_path in models:
        print(model_path)
        onnx_model = load_model(model_path)
        input_names = [node.name for node in onnx_model.graph.input]
        input_tensors = {k: tensors[k] for k in input_names}
        output_tensors = run_segment(onnx_model, input_tensors)

        tensors.update(output_tensors)

    breakpoint()
    print("result")
