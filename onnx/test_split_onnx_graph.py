"""Example of splitting ONNX graph manually

https://github.com/onnx/onnx/blob/main/docs/IR.md
"""

from collections import defaultdict, deque
import enum
from typing import *

import numpy as np
import onnxruntime as ort
from graphviz import Graph
from six import string_types
import tvm
from tvm import relay
import onnx
import onnx.helper
import onnx.shape_inference
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
from onnx.tools import update_model_dims

OPERATORS_OF_INTERST = {
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
DYNAMIC_OPS_TO_DYNAMIC_ARGS = {"Reshape": (1,)}


def get_input_values(
    onnx_model: onnx.ModelProto, tracing_tensors: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    result = {}
    for vinfo_input in onnx_model.graph.input:
        name = vinfo_input.name
        result[name] = tracing_tensors[name]
    return result


def run_model(
    onnx_model: onnx.ModelProto,
    input_values: Dict[str, np.ndarray],
    verbose=False,
) -> Dict[str, np.ndarray]:
    """Run the given model and return the resulting tensors."""
    if not verbose:
        so = ort.SessionOptions()
        so.log_severity_level = 3
    session = ort.InferenceSession(onnx_model.SerializeToString(), sess_options=so)
    result = session.run(None, input_values)

    output_values = {}
    for i, output_node in enumerate(onnx_model.graph.output):
        name = output_node.name
        tensor = result[i]
        output_values[name] = tensor
    return output_values


def run_model_tvm(
    onnx_model: onnx.ModelProto,
    input_values: Dict[str, np.ndarray],
    print_mod=False,
):
    """Run the given model in TVM."""
    # Create TVM session
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape={k: v.shape for k, v in input_values.items()},
        dtype={k: str(v.dtype) for k, v in input_values.items()},
        freeze_params=True,
    )
    input_values.update(params)

    # Print module with tabbed in if appropriate
    if print_mod:
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

    output_values = {}
    for i, output_node in enumerate(onnx_model.graph.output):
        name = output_node.name
        tensor = tvm_result[i]
        output_values[name] = tensor

    return output_values


class SubgraphExtractor:
    def __init__(
        self,
        onnx_model: ModelProto,
        input_values: Dict[str, np.ndarray],
    ) -> None:
        # The onnx model we are dealing with
        self.onnx_model: ModelProto = onnx.shape_inference.infer_shapes(
            onnx_model, data_prop=True
        )

        # The initial set of input values to be used for the onnx model
        self.input_values: Dict[str, np.ndarray] = input_values

        # Map of name of node --> ONNX node
        self.node_map: Dict[str, NodeProto] = {}

        # Map of node edges --> name of node who produce edge
        self.edge_producers: Dict[str, str] = {}

        # Map of node edges --> name of nodes which consume edge
        # We use inner list to maintain topographical ordering
        self.edge_consumers: Dict[str, List[str]] = defaultdict(list)

        # Map of edge name --> value info (type information for nodes/tensors)
        self.value_info_map: Dict[str, ValueInfoProto] = {}

        # Map of edge --> initializer
        self.initializer_map: Dict[str, TensorProto] = {}

        # Map of node name --> topological index
        self.node_name_index: Dict[str, int] = {}

        for i, node in enumerate(self.onnx_model.graph.node):
            name = node.name
            self.node_map[name] = node
            for input_name in node.input:
                self.edge_consumers[input_name].append(name)
            for output_name in node.output:
                self.edge_producers[output_name] = name
            self.node_name_index[name] = i

        for vinfo in self.onnx_model.graph.value_info:
            self.value_info_map[vinfo.name] = vinfo
        for input in self.onnx_model.graph.input:
            self.value_info_map[input.name] = input
        for output in self.onnx_model.graph.output:
            self.value_info_map[output.name] = output

        for tensor in self.onnx_model.graph.initializer:
            self.initializer_map[tensor.name] = tensor

    def _get_tracing_model(self, edges_to_trace: Set[str]):
        """Create a copy of the model modified so the given set of edges are set as outputs."""
        onnx_model = ModelProto()
        onnx_model.CopyFrom(self.onnx_model)
        for edge in edges_to_trace:
            onnx_model.graph.output.append(self.value_info_map[edge])
        return onnx_model

    def _extract_subgraph(
        self,
        node_names: List[str],
        onnx_model: Optional[ModelProto] = None,
        name: str = "",
    ) -> GraphProto:
        """Extract the subgraph consisting of the given nodes.

        If onnx_model is None, uses self.onnx_model.
        """
        if onnx_model is None:
            onnx_model = self.onnx_model

        # Note node_names must be in topological order
        # Information needed to construct new graph.
        nodes: List[NodeProto] = []
        inputs: List[ValueInfoProto] = []
        outputs: List[ValueInfoProto] = []
        initializer: List[TensorProto] = []

        possible_outputs: List[str] = []
        possible_inputs: List[str] = []

        # Local versions of self.edge_producer/consumers
        local_edge_producers = {}
        local_edge_consumers = defaultdict(list)
        all_edges = set()

        # Nodes must be sorted in original topological order, we do so based
        node_names.sort(key=lambda node_name: self.node_name_index[node_name])

        # Here we append into `nodes` and construct local_edge_* vars
        for name in node_names:
            node = self.node_map[name]

            # Create the new protobuf for node
            node_proto = NodeProto()
            node_proto.CopyFrom(node)
            nodes.append(node_proto)

            for node_input in node.input:
                local_edge_consumers[node_input].append(name)
                possible_inputs.append(node_input)
                all_edges.add(node_input)
            for node_output in node.output:
                local_edge_producers[node_output] = name
                possible_outputs.append(node_output)
                all_edges.add(node_output)

        # Prevent duplicate inputs
        inputs_added = set()
        # Here we can determine the inputs and outputs of our graph
        # inputs are names with no producers and not constants (initializers)
        # outputs are names where local consumers != global consumers OR were outputs in the original graph
        for name in possible_inputs:
            if (
                name not in local_edge_producers
                and name not in self.initializer_map
                and name not in inputs_added
            ):
                inputs_added.add(name)
                vinfo = ValueInfoProto()
                vinfo.CopyFrom(self.value_info_map[name])
                inputs.append(vinfo)

        global_output_names = {vinfo.name for vinfo in self.onnx_model.graph.output}
        for name in possible_outputs:
            if (
                len(local_edge_consumers[name]) != len(self.edge_consumers[name])
                or name in global_output_names
            ):
                vinfo = ValueInfoProto()
                vinfo.CopyFrom(self.value_info_map[name])
                outputs.append(vinfo)

        # Initializers just get copied over (new initializers might get copied later)
        for edge in all_edges:
            if edge in self.initializer_map:
                tensor_proto = TensorProto()
                tensor_proto.CopyFrom(self.initializer_map[edge])
                initializer.append(tensor_proto)

        return onnx.helper.make_graph(
            nodes,
            name,
            inputs,
            outputs,
            initializer,
        )

    def _get_sections(
        self, operators_of_interest: Set[str], max_ops_of_interest: int = 1
    ) -> List[List[str]]:
        """Does a BFS from an anchor node, stopping search when it finds enough ops of interest."""
        sections = []
        visited = set()
        for anchor_node in self.onnx_model.graph.node:
            cur_section = []
            ops_of_interest = 0
            search_space = deque()
            search_space.append(anchor_node.name)
            while len(search_space) > 0 and ops_of_interest < max_ops_of_interest:
                cur_node_name = search_space.popleft()
                if cur_node_name in visited:
                    continue
                visited.add(cur_node_name)

                cur_node = self.node_map[cur_node_name]
                for output_edge in cur_node.output:
                    for node_name in self.edge_consumers[output_edge]:
                        search_space.append(node_name)
                for input_edge in cur_node.input:
                    if input_edge in self.edge_producers:
                        search_space.append(self.edge_producers[input_edge])

                cur_section.append(cur_node.name)
                if cur_node.op_type in operators_of_interest:
                    ops_of_interest += 1

            if len(cur_section) > 0:
                sections.append(cur_section)

        return sections

    def _infer_shapes(
        self, input_onnx_model: ModelProto, tracing_tensors: Dict[str, np.ndarray]
    ):
        """Realize input shapes for the given input onnx model based on data in tracing_tensors."""
        onnx_model = ModelProto()
        onnx_model.CopyFrom(input_onnx_model)
        new_vinfos = {}

        # Construct new vinfos for each input based on tracing tensor
        for old_vinfo in list(input_onnx_model.graph.input):
            name = old_vinfo.name
            input_value = tracing_tensors[name]

            # TODO: handle other types besides tensors, e.g. sequence
            new_type_proto = onnx.helper.make_tensor_type_proto(
                elem_type=self.value_info_map[name].type.tensor_type.elem_type,
                shape=input_value.shape,
            )
            vinfo = onnx.helper.make_value_info(name, new_type_proto)
            new_vinfos[name] = vinfo

        for input_vinfo in onnx_model.graph.input:
            input_vinfo.CopyFrom(new_vinfos[input_vinfo.name])

        # With input tensor shapes realized, this should solve much more!
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)
        return onnx_model

    def _remove_dynamism(
        self,
        input_onnx_model: ModelProto,
        tracing_tensors: Dict[str, np.ndarray],
        dynamic_ops: Dict[str, List[int]] = DYNAMIC_OPS_TO_DYNAMIC_ARGS,
    ) -> onnx.ModelProto:
        """Removes dynamism by setting certain inputs as constant if they feed into a dynamic op."

        Dynamic ops is a map of operator name to list of all indices which should be frozen.
        """
        onnx_model = ModelProto()
        onnx_model.CopyFrom(input_onnx_model)

        inputs_to_freeze = set()
        for node in onnx_model.graph.node:
            if node.op_type in dynamic_ops.keys():
                bad_arg_pos = dynamic_ops[node.op_type]
                for arg_pos in bad_arg_pos:
                    input_edge = node.input[arg_pos]
                    if input_edge in tracing_tensors.keys():
                        inputs_to_freeze.add(input_edge)

        for k in inputs_to_freeze:
            vinfo = self.value_info_map[k]

            # TODO: handle other types besides tensors, e.g. sequence
            tensor_proto = onnx.helper.make_tensor(
                name=k,
                data_type=vinfo.type.tensor_type.elem_type,
                dims=tracing_tensors[k].shape,
                vals=tracing_tensors[k].flatten().tolist(),
            )
            onnx_model.graph.initializer.append(tensor_proto)

            # Now it is initializer, remove from inputs
            for i, input_node in enumerate(list(onnx_model.graph.input)):
                if input_node.name == k:
                    onnx_model.graph.input.pop(i)
                    break

        return onnx_model

    def _sections_to_models_and_tensors(
        self, sections: List[List[str]]
    ) -> Tuple[List[ModelProto], Dict[str, np.ndarray]]:
        # Grab new models and list of edges to trace by running tracing model
        new_models = []
        edges_to_trace = set()
        for section in sections:
            subgraph = self._extract_subgraph(section)
            new_onnx_model = onnx.helper.make_model(subgraph)
            # Get the right opset!
            for opset_i in range(len(new_onnx_model.opset_import)):
                new_onnx_model.opset_import[opset_i].CopyFrom(
                    self.onnx_model.opset_import[opset_i]
                )
            new_models.append(new_onnx_model)
            for output in new_onnx_model.graph.input:
                edges_to_trace.add(output.name)

        # List of output tensors around each op of interest in the model
        tracing_model = self._get_tracing_model(edges_to_trace)

        # All tensors at frontier of an operator of interest
        tracing_tensors = run_model(tracing_model, self.input_values)
        tracing_tensors.update(self.input_values)

        finalized_models = []
        for new_onnx_model in new_models:
            new_onnx_model = self._remove_dynamism(new_onnx_model, tracing_tensors)
            new_onnx_model = self._infer_shapes(new_onnx_model, tracing_tensors)
            finalized_models.append(new_onnx_model)
        return finalized_models, tracing_tensors

    def extract_subgraphs(
        self,
        operators_of_interest: Set[str] = OPERATORS_OF_INTERST,
        max_operators_of_interest: int = 1,
    ):
        sections = self._get_sections(
            operators_of_interest, max_ops_of_interest=max_operators_of_interest
        )

        return self._sections_to_models_and_tensors(sections)

    def extract_single_operators(
        self, operators_of_interest: Set[str] = OPERATORS_OF_INTERST
    ):
        sections = []
        for node in self.onnx_model.graph.node:
            if node.op_type in operators_of_interest:
                sections.append([node.name])
        return self._sections_to_models_and_tensors(sections)


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


def verify_models(model: ModelProto, tracing_tensors: Dict[str, np.ndarray]):
    input_values = get_input_values(model, tracing_tensors)
    # Just make sure both can run
    run_model(model, input_values)
    run_model_tvm(model, input_values)


def run_examination(
    input_shapes: Dict[str, List[int]],
    input_dtypes: Dict[str, str],
    model_path: str,
):
    tensors = {
        name: np.zeros(input_shapes[name]).astype(input_dtypes[name])
        for name in input_shapes.keys()
    }
    onnx_model = onnx.load(model_path)
    extractor = SubgraphExtractor(onnx_model, tensors)

    print(f"Handling {model_path}")
    (
        single_operator_models,
        single_operator_tensors,
    ) = extractor.extract_single_operators()

    for i, model in enumerate(single_operator_models):
        print(f"\tVerifying single model {i + 1} / {len(single_operator_models)}")
        verify_models(model, single_operator_tensors)

    subgraph, subgraph_tensors = extractor.extract_subgraphs()
    for i, model in enumerate(subgraph):
        print(f"\tVerifying single model {i + 1} / {len(subgraph)}")
        verify_models(model, subgraph_tensors)
    print()


if __name__ == "__main__":
    try_bertsquad()
    try_ssd()
    try_yolov2()
    try_shufflenet()
    try_resnet()