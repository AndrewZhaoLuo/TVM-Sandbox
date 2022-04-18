"""Example of rewriting model inputs. May or may not work depending on other components of graph."""

from typing import Dict

import numpy as np

import onnx
import onnx.checker
import onnx.shape_inference


class OnnxInputShapeChanger:
    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model: onnx.ModelProto = onnx.shape_inference.infer_shapes(
            onnx_model, data_prop=True
        )
        onnx_model_copy = onnx.ModelProto()
        onnx_model_copy.CopyFrom(onnx_model)
        self.onnx_model = onnx_model_copy

        self.old_vinfo_map = self._get_vinfo_map(self.onnx_model)

        # Erase type annotations for intermediate nodes
        self.old_outputs = []
        while len(self.onnx_model.graph.output) > 0:
            self.old_outputs.append(self.onnx_model.graph.output.pop().name)
        while len(self.onnx_model.graph.value_info) > 0:
            self.onnx_model.graph.value_info.pop()

    def _get_vinfo_map(
        self, onnx_model: onnx.ModelProto
    ) -> Dict[str, onnx.ValueInfoProto]:
        value_info_map: Dict[str, onnx.ValueInfoProto] = {}
        for input in onnx_model.graph.input:
            value_info_map[input.name] = input
        for output in onnx_model.graph.output:
            value_info_map[output.name] = output
        for vinfo in onnx_model.graph.value_info:
            value_info_map[vinfo.name] = vinfo
        return value_info_map

    def rewrite(self, tracing_tensors: Dict[str, np.ndarray]) -> onnx.ModelProto:
        """Realize input shapes for the given input onnx model based on data in tracing_tensors.

        :params tracing_tensors: A map of edge names to tensors we have type information of.

        :returns: A new onnx model with shape information realized.
        """
        onnx_model = onnx.ModelProto()
        onnx_model.CopyFrom(self.onnx_model)
        new_vinfos_inputs = {}

        # Construct new vinfos for each input based on tracing tensor
        for old_vinfo in list(onnx_model.graph.input):
            name = old_vinfo.name
            input_value = tracing_tensors[name]

            # TODO: handle other types besides tensors, e.g. sequence
            new_type_proto = onnx.helper.make_tensor_type_proto(
                elem_type=self.old_vinfo_map[name].type.tensor_type.elem_type,
                shape=input_value.shape,
            )
            vinfo = onnx.helper.make_value_info(name, new_type_proto)
            new_vinfos_inputs[name] = vinfo

        for input_vinfo in onnx_model.graph.input:
            input_vinfo.CopyFrom(new_vinfos_inputs[input_vinfo.name])

        # With input tensor shapes realized, this should solve much more!
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model, data_prop=True)

        # re-add outputs by moving from value_infos --> outputs
        new_vinfo_map = self._get_vinfo_map(onnx_model)
        for output_name in self.old_outputs:
            onnx_model.graph.output.append(new_vinfo_map[output_name])
            onnx_model.graph.value_info.remove(new_vinfo_map[output_name])

        return onnx_model


if __name__ == "__main__":
    onnx_model = onnx.load("models/mobilenetv2-7-2.onnx")

    # Map of onnx input nodes --> tensor to trace/set model with
    input_dict = {"input": np.zeros([1, 3, 32, 32], dtype="float32")}
    rewriter = OnnxInputShapeChanger(onnx_model)
    new_model = rewriter.rewrite(input_dict)

    # Model may not be totally valid esp. with dynamic ops but should be fine to run/ingest
    # onnx.checker.check_model(new_model, full_check=True)
    onnx.save(new_model, "result.onnx")
