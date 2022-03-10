"""Example of splitting ONNX graph manually

https://github.com/onnx/onnx/blob/main/docs/IR.md
"""

from collections import defaultdict, deque

import numpy as np

import onnx.utils
from onnx import shape_inference
from onnx.tools import update_model_dims

OP_TYPES_OF_INTEREST = {"Conv", "MatMul"}


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


def split_model(onnx_model, op_types_of_interest=OP_TYPES_OF_INTEREST):
    extractor = onnx.utils.Extractor(onnx_model)

    # Map of name of node --> ONNX node
    node_map = {}

    # Constant tensors
    default_tensors = {t.name for t in onnx_model.graph.initializer}

    # Map of names --> names of nodes who consume name
    consumer_map = defaultdict(set)

    for node in onnx_model.graph.node:
        node_map[node.name] = node
        for input_name in node.input:
            consumer_map[input_name].add(node.name)

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
                if ops_of_interest > 1:
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


if __name__ == "__main__":
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
    model_segments = split_model(onnx_model)
    for i, segment in enumerate(model_segments):
        onnx.save(segment, f"bertsquad-segment{i}.onnx")

    onnx_model = load_model("models/ssd-10.onnx")
    model_segments = split_model(onnx_model)
    for i, segment in enumerate(model_segments):
        onnx.save(segment, f"ssd-segment{i}.onnx")