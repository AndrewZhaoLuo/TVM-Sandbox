"""
For Nodes of interest, adds output taps to the input and output taps.
"""

from collections import defaultdict, deque
from typing import *

import numpy as np
import onnxruntime as ort
from graphviz import Graph
from six import string_types

import onnx
import onnx.helper
import onnx.shape_inference
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto
from onnx.tools import update_model_dims

OPERATORS = {"Conv", "MatMul"}
DYNAMIC_OPS = {"Reshape": (1,)}


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


class SubgraphExtractor:
    def __init__(
        self,
        onnx_model: ModelProto,
        input_values: Dict[str, np.ndarray],
        operators_of_interest: Set[str] = OPERATORS,
    ) -> None:
        # This doesn't work too well with some models. Future Idea: run model to get shapes
        # Set shapes so each component is able to be calculated with real shapes
        onnx_model = update_inputs_outputs_dims(
            onnx_model,
            {k: v.shape for k, v in input_values.items()},
            {},
        )
        self.onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        self.input_values = input_values
        self.operators_of_interest = set(operators_of_interest)

        # Map of name of node --> ONNX node
        self.node_map: Dict[str, NodeProto] = {}

        # Map of node edges --> name of node who produce edge
        self.edge_producers: Dict[str, str] = {}

        # Map of node edges --> name of nodes which consume edge
        # We use inner list to maintain topographical ordering
        self.edge_consumers: Dict[str, List[str]] = defaultdict(list)

        # List of nodes with interesting operators in proper order
        self.nodes_of_interest: List[str] = []
        for node in self.onnx_model.graph.node:
            name = node.name
            self.node_map[name] = node
            for input_name in node.input:
                self.edge_consumers[input_name].append(name)
            for output_name in node.output:
                self.edge_producers[output_name] = name
            if node.op_type in self.operators_of_interest:
                self.nodes_of_interest.append(name)

        # Map of edge name --> value info (type information for nodes/tensors)
        self.value_info_map: Dict[str, ValueInfoProto] = {
            vinfo.name: vinfo for vinfo in self.onnx_model.graph.value_info
        }
        for input in self.onnx_model.graph.input:
            self.value_info_map[input.name] = input
        for output in self.onnx_model.graph.output:
            self.value_info_map[output.name] = output

        # Map of edge --> initializer
        self.initializer_map: Dict[str, TensorProto] = {
            tensor.name: tensor for tensor in self.onnx_model.graph.initializer
        }

    def _get_tracing_model(self):
        # Create copy of the model
        onnx_model = ModelProto()
        onnx_model.CopyFrom(self.onnx_model)

        # For every node of interest, set inputs and outputs to nodes as graph outputs
        # To collect info in tracing the model.
        for operator in onnx_model.graph.node:
            if operator.op_type in self.operators_of_interest:
                operator_inputs = list(operator.input)
                operator_outputs = list(operator.output)
                new_graph_output_names = operator_inputs + operator_outputs
                for new_output_name in new_graph_output_names:
                    onnx_model.graph.output.append(self.value_info_map[new_output_name])
        return onnx_model

    def _run_model(
        self,
        onnx_model: onnx.ModelProto,
        input_values: Dict[str, np.ndarray],
        verbose=False,
    ) -> Dict[str, np.ndarray]:
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

    def _extract_subgraph(self, node_names: List[str], name: str = "") -> GraphProto:
        # Note node_names must be in topological order
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

        # Here we can determine the inputs and outputs of our graph
        # inputs are names with no producers and not constants (initializers)
        # outputs are names where local consumers != global consumers OR were outputs in the original graph
        for name in possible_inputs:
            if name not in local_edge_producers and name not in self.initializer_map:
                vinfo = ValueInfoProto()
                vinfo.CopyFrom(self.value_info_map[name])
                inputs.append(vinfo)
        for name in possible_outputs:
            if len(local_edge_consumers[name]) != len(
                self.edge_consumers[name]
            ) or name in [vinfo.name for vinfo in self.onnx_model.graph.output]:
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

    def _get_sections(self) -> List[List[str]]:
        all_sections = []
        visited_nodes = set()
        nodes_of_interest_set = set(self.nodes_of_interest)

        # Note self.nodes_of_interest is in topological order.
        for anchor_node_name in self.nodes_of_interest:
            search_space = deque([anchor_node_name])
            cur_section = []
            while len(search_space) > 0:
                cur_node_name = search_space.popleft()
                if cur_node_name in visited_nodes:
                    continue
                if cur_node_name in nodes_of_interest_set:
                    if cur_node_name != anchor_node_name:
                        continue

                cur_node = self.node_map[cur_node_name]
                visited_nodes.add(cur_node_name)
                cur_section.append(cur_node_name)

                # Expand search space
                for input_edge in cur_node.input:
                    if input_edge in self.edge_producers:
                        search_space.appendleft(self.edge_producers[input_edge])
                for output_edge in cur_node.output:
                    for node_name in self.edge_consumers[output_edge]:
                        search_space.appendleft(node_name)
            all_sections.append(cur_section)
        return all_sections

    def _infer_shapes(
        self, input_onnx_model: ModelProto, input_values: Dict[str, np.ndarray]
    ):
        onnx_model = ModelProto()
        onnx_model.CopyFrom(input_onnx_model)
        new_vinfos = {}
        for name, input_value in input_values.items():
            new_type_proto = onnx.helper.make_tensor_type_proto(
                elem_type=self.value_info_map[name].type.tensor_type.elem_type,
                shape=input_value.shape,
            )
            vinfo = onnx.helper.make_value_info(name, new_type_proto)
            new_vinfos[name] = vinfo

        assert len(onnx_model.graph.input) == len(new_vinfos)
        for input_vinfo in onnx_model.graph.input:
            input_vinfo.CopyFrom(new_vinfos[input_vinfo.name])
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        return onnx_model

    def _remove_dynamism(
        self,
        input_onnx_model: ModelProto,
        input_values: Dict[str, np.ndarray],
        dynamic_ops: Dict[str, List[int]] = DYNAMIC_OPS,
    ) -> onnx.ModelProto:
        """Runs the onnx model while also mutating it to make certain dynamic ops non-dynamic."""
        # Get list of inputs which should be frozen (set as constants)
        # Because they feed into a dynamic op's argument which causes dynamism

        onnx_model = ModelProto()
        onnx_model.CopyFrom(input_onnx_model)

        inputs_to_freeze = set()
        for node in onnx_model.graph.node:
            if node.op_type in dynamic_ops.keys():
                bad_arg_pos = dynamic_ops[node.op_type]
                for arg_pos in bad_arg_pos:
                    input_edge = node.input[arg_pos]
                    if input_edge in input_values.keys():
                        inputs_to_freeze.add(input_edge)
                    """
                    # This doesn't work super well since it simplifies too much
                    # Solve which edges might be dependent for dynamism
                    visited_edges = set()
                    search_space_edges = [input_edge]
                    while len(search_space_edges) > 0:
                        cur_edge_name = search_space_edges.pop()
                        if cur_edge_name in visited_edges:
                            continue

                        visited_edges.add(cur_edge_name)
                        if cur_edge_name in input_values.keys():
                            inputs_to_freeze.add(cur_edge_name)

                        parent_node_name = self.edge_producers.get(cur_edge_name, None)
                        if parent_node_name is not None:
                            parent_node = self.node_map[parent_node_name]
                            for input_edge in parent_node.input:
                                search_space_edges.append(input_edge)
                    """

        for k in inputs_to_freeze:
            vinfo = self.value_info_map[k]

            # TODO: handle other types besides tensors, e.g. sequence
            tensor_proto = onnx.helper.make_tensor(
                name=k,
                data_type=vinfo.type.tensor_type.elem_type,
                dims=input_values[k].shape,
                vals=input_values[k].flatten().tolist(),
            )
            onnx_model.graph.initializer.append(tensor_proto)

            # Now it is initializer, remove from inputs
            for i, input_node in enumerate(list(onnx_model.graph.input)):
                if input_node.name == k:
                    onnx_model.graph.input.pop(i)
                    break

        return onnx_model

    def _get_input_tensors(
        onnx_model: ModelProto, tracing_tensors: Dict[str, np.ndarray]
    ):
        result = {}
        for input_edge in onnx_model.graph.input:
            if input_edge in tracing_tensors:
                result[input_edge] = tracing_tensors
            elif input_edge:
                pass
        return {
            vinfo.name: tracing_tensors[vinfo.name] for vinfo in onnx_model.graph.input
        }

    def extract_full_graph(self):
        # List of output tensors around each op of interest in the model
        tracing_model = self._get_tracing_model()

        # All tensors at frontier of an operator of interest
        tracing_tensors = self._run_model(tracing_model, self.input_values)
        tracing_tensors.update(self.input_values)

        sections = self._get_sections()
        for i, section in enumerate(sections):
            print(i)
            subgraph = self._extract_subgraph(section)
            new_onnx_model = onnx.helper.make_model(subgraph)
            # Get the right opset!
            for j in range(len(new_onnx_model.opset_import)):
                new_onnx_model.opset_import[j].CopyFrom(self.onnx_model.opset_import[j])

            new_input_tensors = {
                vinfo.name: tracing_tensors[vinfo.name]
                for vinfo in new_onnx_model.graph.input
            }
            new_onnx_model = self._remove_dynamism(new_onnx_model, new_input_tensors)
            new_input_tensors = {
                vinfo.name: tracing_tensors[vinfo.name]
                for vinfo in new_onnx_model.graph.input
            }
            new_onnx_model = self._infer_shapes(new_onnx_model, new_input_tensors)
            onnx.save(new_onnx_model, f"{i}.onnx")

    """
    def extract_ops_of_interest_graph(self, input_values: Dict[str, np.ndarray]):
        tracing_model = self._get_tracing_model()
        tracing_tensors = self._run_model(tracing_model, input_values)
        for i, node_name in enumerate(self.nodes_of_interest):
            graph = self._extract_subgraph([node_name])
            new_model_proto = onnx.helper.make_model(graph)
            onnx.save(new_model_proto, f"{i}.onnx")
    """


if __name__ == "__main__":
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
    tensors = {
        name: np.zeros(input_shapes[name]).astype(input_dtypes[name])
        for name in input_shapes.keys()
    }
    onnx_model = onnx.load("models/bertsquad-8.onnx")
    extractor = SubgraphExtractor(onnx_model, tensors)
    extractor.extract_full_graph()
    onnx.save(extractor.onnx_model, "_0.onnx")
